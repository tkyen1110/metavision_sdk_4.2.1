/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_PIPELINE_BUILDER_H
#define METAVISION_SDK_ML_PIPELINE_BUILDER_H

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/ml/events/event_tracked_box.h"
#include "metavision/sdk/ml/components/slicer.h"
#include "metavision/sdk/ml/components/preprocessing.h"
#include "metavision/sdk/ml/components/event_provider.h"
#include "metavision/sdk/ml/components/object_detector.h"
#include "metavision/sdk/ml/components/data_association.h"
#include "metavision/sdk/ml/components/display.h"

namespace Metavision {
template<typename EventType = EventCD>
using DataAssociationBboxTrackedBox = DataAssociation<EventType, EventBbox, EventTrackedBox>;

/// Structure to store parameters required to build and configure the detection and tracking pipeline
struct DTPipeline_params {
    // Pipeline options
    /// Delta of time between two updates of output
    /// @note This parameter changes the behavior of the pipeline
    timestamp pipeline_delta_t_ = 10000;

    // Input options
    /// File name with the input events
    std::string record_file_ = "";
    /// Timestamp at which the pipeline should start
    timestamp start_ts_ = 0;
    /// Timestamp at which the pipeline should stop
    timestamp end_ts_ = std::numeric_limits<timestamp>::max();

    // Output options
    /// File name to store detection boxes
    std::string output_detections_filename_ = "";
    /// File name to store track information
    std::string output_tracks_filename_ = "";
    /// File name to store video of the pipeline output
    std::string output_video_filename_ = "";

    // Display options
    /// Activates the display rendering
    bool display_ = false;
    /// Refresh rate
    int fps_ = 25;

    // Geometric options
    /// Region Of Interest set on the camera side
    std::vector<int> camera_roi_;
    /// Transpose the input event format
    bool transpose_input_ = false;
    /// Flip the events over X-axis
    bool flipX_ = false;
    /// Flip the events over Y-axis
    bool flipY_ = false;

    // Noise Filtering options
    /// Sets the noise filtering algorithm, available options are "trail" or "stc"
    std::string noise_filtering_type_ = "trail";
    /// Threshold provided for the noise filter algorithm
    timestamp noise_filtering_threshold_ = 10000;

    // Object Detector options
    /// Folder from which the neural network is loaded
    std::string object_detector_dir_ = "";
    /// Minimal confidence for box detection
    float detector_confidence_threshold_ = 0.f;
    /// Threshold on Intersection Over Union value for checking box similarity
    float detector_NMS_IOU_threshold_ = 0.f;
    /// Device running the neural network
    std::string ml_runtime_ = "gpu";
    /// Neural Network input frame's width
    int network_input_width_ = 0;
    /// Neural Network input frame's height
    int network_input_height_ = 0;

    // Data Association options
    /// Weight to compute weighted track confidence
    float detection_merge_weight_ = 0.7f;
    /// Time without detection above which a track has to be deleted
    timestamp deletion_time_ = 100000;
    /// Maximal IOU between two tracks to be considered as different
    float max_iou_inter_track_ = 0.5f;
    /// Threshold on IOU value to match two boxes
    float iou_to_match_a_detection_ = 0.2f;
    /// Threshold above which a detection is considered ambiguous.
    /// It means that, a detection is discarded if several tracks match with an IOU
    //// above this max_iou_for_one_det_to_many_tracks_ value.
    float max_iou_for_one_det_to_many_tracks_ = 0.5f;
    /// Use a descriptor to check if two boxes are matching
    bool use_descriptor_ = false;
    /// Number of consecutive detections before creating a new track
    int number_of_consecutive_detections_to_create_a_new_track_ = 1;
    /// Delta of time used to generate the timesurface
    timestamp timesurface_delta_t_ = 200000;
    /// Update tracklets between detections
    bool do_not_update_tracklets_between_detections_;

    // Event Rate options
    bool limit_event_rate_ = false;
    uint32_t max_event_rate_;
};

/// @brief Detection and Tracking pipeline
/// @tparam Event Event type consumed by the pipeline
template<typename Event>
class DTPipeline {
public:
    /// @brief Constructs a Detection and Tracking Pipeline
    /// @param filename_tracks Filename to store the serialized track objects
    /// @param filename_detections Filename to store the serialized detections
    DTPipeline(std::string filename_tracks = "", std::string filename_detections = "") :
        output_filename_tracks_(filename_tracks), output_filename_detections_(filename_detections) {
        if (output_filename_tracks_ != "") {
            output_stream_tracks_.reset(new std::ofstream());
            output_stream_tracks_->open(output_filename_tracks_);
        }

        if (output_filename_detections_ != "") {
            output_stream_detections_.reset(new std::ofstream());
            output_stream_detections_->open(output_filename_detections_);
        }
    }

    /// @brief Builds and runs the pipeline
    /// @param params Parameters to configure the execution
    void build_and_run(const DTPipeline_params &params) {
        const timestamp pipeline_delta_t = params.pipeline_delta_t_;

        // create event provider
        ev_provider_ = create_EventProvider(params);
        assert(ev_provider_ != nullptr);
        int width_cam                = ev_provider_->get_width();
        int height_cam               = ev_provider_->get_height();
        bool do_transpose            = params.transpose_input_;
        int width_after_geompreproc  = do_transpose ? height_cam : width_cam;
        int height_after_geompreproc = do_transpose ? width_cam : height_cam;

        if (params.limit_event_rate_) {
            ev_provider_->set_event_rate_limit(params.max_event_rate_);
        }

        // create geometric preproc
        geom_preproc_          = create_GeometricPreprocessing(params, width_cam, height_cam);
        publisher_geom_filter_ = create_EventSlicer(params, geom_preproc_.get());

        // create noise preproc
        noise_preproc_ = create_NoiseFilterPreprocessing(params, width_after_geompreproc, height_after_geompreproc);
        publisher_noise_filter_ = create_EventSlicer(params, noise_preproc_.get());

        // create object detector
        object_detector_ = create_ObjectDetector(params, width_after_geompreproc, height_after_geompreproc);
        if (object_detector_->get_accumulation_time() % pipeline_delta_t != 0) {
            std::ostringstream oss;
            oss << "Error: Object detector accumulation time (" << object_detector_->get_accumulation_time()
                << ") must be a multiple of pipeline delta_t (" << pipeline_delta_t << ")" << std::endl;
            throw std::logic_error(oss.str());
        }

        // create data associator
        data_associator_ = create_DataAssociator(params, width_after_geompreproc, height_after_geompreproc);

        // create display generator (for display)
        if (params.display_ || (params.output_video_filename_ != "")) {
            display_.reset(new DetectionAndTrackingDisplay<Event>(width_after_geompreproc, height_after_geompreproc,
                                                                  pipeline_delta_t, params.fps_,
                                                                  params.output_video_filename_, params.display_));
            display_->set_ui_keys(std::bind(&DTPipeline<Event>::ui_key_bindings, this, std::placeholders::_1));
        }
        if (!params.display_) {
            MV_SDK_LOG_INFO()
                << "Display window disabled by default. Use  --display   to enable display during processing"
                << std::endl;
        }

        connect_callbacks();
        run();
        clean_stop();
    }

private:
    /// @brief Builds an EventProvider object
    /// @param params Parameters to build the event provider
    /// @return An EventProvider object
    static std::unique_ptr<EventProviderBase> create_EventProvider(const DTPipeline_params &params) {
        const std::string input = params.record_file_;

        if (input == "") {
            // live cam
            return std::unique_ptr<EventProviderBase>(new EventProviderRaw());
        }

        timestamp start_ts = params.start_ts_;
        timestamp end_ts   = params.end_ts_;

        std::unique_ptr<EventProviderBase> event_provider;
        if (bfs::path(input).extension() == ".raw") {
            event_provider.reset(new EventProviderRaw(input));
        } else if (bfs::path(input).extension() == ".dat") {
            event_provider.reset(new EventProviderDat(input));
        } else {
            std::ostringstream oss;
            oss << "Wrong extension for input sequence: " << input << std::endl;
            throw std::logic_error(oss.str());
        }

        event_provider->set_start_ts(start_ts);
        event_provider->set_end_ts(end_ts);
        return event_provider;
    }

    /// @brief Builds an ObjectDetector component
    /// @param params Parameters to build the ObjectDetector object
    /// @param events_width Sensor's width
    /// @param events_height Sensor's height
    /// @return an ObjectDetector
    static std::unique_ptr<ObjectDetectorBaseT<Event>> create_ObjectDetector(const DTPipeline_params &params,
                                                                             int events_width, int events_height) {
        std::unique_ptr<ObjectDetectorBaseT<Event>> object_detector;
        std::string model_dir = params.object_detector_dir_;
        std::string runtime   = params.ml_runtime_;
        int network_width     = params.network_input_width_;
        int network_height    = params.network_input_height_;

        object_detector.reset(new ObjectDetectorT<EventCD>(model_dir, runtime, events_width, events_height,
                                                           network_width, network_height));
        if (params.detector_confidence_threshold_ != 0.f) {
            object_detector->set_detection_threshold(params.detector_confidence_threshold_);
        }
        if (params.detector_NMS_IOU_threshold_ != 0.f) {
            object_detector->set_iou_threshold(params.detector_NMS_IOU_threshold_);
        }
        return object_detector;
    }

    /// @brief Builds a DataAssociation object
    /// @param params Parameters to build the DataAssociation object
    /// @param width Sensor's width
    /// @param height Sensor's height
    /// @return A DataAssociation object
    static std::unique_ptr<DataAssociationBboxTrackedBox<Event>> create_DataAssociator(const DTPipeline_params &params,
                                                                                       int width, int height) {
        std::unique_ptr<DataAssociationBboxTrackedBox<Event>> data_associator;
        const float detection_merge_weight             = params.detection_merge_weight_;
        const timestamp deletion_time                  = params.deletion_time_;
        const float max_iou_inter_track                = params.max_iou_inter_track_;
        const float iou_to_match_a_detection           = params.iou_to_match_a_detection_;
        const float max_iou_for_one_det_to_many_tracks = params.max_iou_for_one_det_to_many_tracks_;
        bool use_descriptor                            = params.use_descriptor_;
        const int number_of_consecutive_detections_to_create_a_new_track =
            params.number_of_consecutive_detections_to_create_a_new_track_;
        timestamp timesurface_deltat_t           = params.timesurface_delta_t_;
        bool update_tracklets_between_detections = !params.do_not_update_tracklets_between_detections_;

        data_associator.reset(new DataAssociationBboxTrackedBox<Event>(
            detection_merge_weight, deletion_time, max_iou_inter_track, iou_to_match_a_detection,
            max_iou_for_one_det_to_many_tracks, use_descriptor, number_of_consecutive_detections_to_create_a_new_track,
            width, height, timesurface_deltat_t, update_tracklets_between_detections));
        return data_associator;
    }

    /// @brief Builds a PreprocessingBase object
    /// @param params Parameters to build Preprocessing object
    /// @param width Sensor's width
    /// @param height Sensor's height
    /// @return A Preprocessing component
    static std::unique_ptr<PreprocessingBase<Event>> create_GeometricPreprocessing(const DTPipeline_params &params,
                                                                                   int width, int height) {
        std::unique_ptr<GeometricPreprocessing<Event>> geom_preproc;
        geom_preproc.reset(new GeometricPreprocessing<Event>(width, height));
        if (!params.camera_roi_.empty()) {
            const std::vector<int> roi = params.camera_roi_;
            assert(roi.size() == 4);
            geom_preproc->use_roi(roi[0], roi[1], roi[2], roi[3]);
        }
        bool do_transpose = params.transpose_input_;
        bool do_flip_x    = params.flipX_;
        bool do_flip_y    = params.flipY_;
        geom_preproc->use_transpose_flipxy(do_transpose, do_flip_x, do_flip_y);
        return geom_preproc;
    }

    /// @brief Builds a NoiseFilterPreprocessing object
    /// @param params Parameters to build the NoiseFilterPreprocessing object
    /// @param width Sensor's width
    /// @param height Sensor's height
    /// @return A NoiseFilterPreprocessing component
    static std::unique_ptr<PreprocessingBase<Event>> create_NoiseFilterPreprocessing(const DTPipeline_params &params,
                                                                                     int width, int height) {
        timestamp noise_filtering_thr    = params.noise_filtering_threshold_;
        std::string noise_filtering_type = params.noise_filtering_type_;
        if (noise_filtering_thr == 0) {
            return nullptr;
        }
        if (noise_filtering_type == "trail") {
            std::unique_ptr<NoiseFilterPreprocessing<Event, TrailFilterAlgorithm>> noise_preproc(
                new NoiseFilterPreprocessing<Event, TrailFilterAlgorithm>(width, height, noise_filtering_thr));
            return noise_preproc;
        } else if (noise_filtering_type == "stc") {
            std::unique_ptr<NoiseFilterPreprocessing<Event, SpatioTemporalContrastAlgorithm>> noise_preproc(
                new NoiseFilterPreprocessing<Event, SpatioTemporalContrastAlgorithm>(width, height,
                                                                                     noise_filtering_thr));
            return noise_preproc;
        } else {
            std::ostringstream oss;
            oss << "Error: unsupported type of noise filtering: " << noise_filtering_type << std::endl;
            throw std::logic_error(oss.str());
        }
        return nullptr;
    }

    /// @brief Builds an EventSlicer object used to cadence the pipeline
    /// @param params Parameters to build the EventSlicer object
    /// @param preproc Preprocessing (noise filter)
    /// @return an EventSlicer component
    static std::unique_ptr<Slicer<Event>> create_EventSlicer(const DTPipeline_params &params,
                                                             PreprocessingBase<Event> *preproc) {
        timestamp pipeline_delta_t = params.pipeline_delta_t_;
        std::unique_ptr<Slicer<Event>> publisher;

        if (preproc == nullptr) {
            publisher.reset(new Slicer<Event>(pipeline_delta_t));
        } else {
            publisher.reset(new Slicer<Event>(pipeline_delta_t, preproc->get_preprocessing_callback()));
        }
        return publisher;
    }

    /// @brief Handles keyboard events
    /// @param key Key that has been pressed
    void ui_key_bindings(int key) {
        switch (key) {
        case 'q':
        case 27: // escape
            ev_provider_->stop();
            MV_SDK_LOG_INFO() << "!! STOP !!";
            break;
        case 255:
            break;
        default:
            break;
        }
    }

    using EventBoxConsumerCallback =
        std::function<void(const EventBbox *begin, const EventBbox *end, timestamp ts, bool is_valid)>;

    /// @brief Creates a function to be called on generated bbox
    /// @return Function handling new bboxes
    EventBoxConsumerCallback get_bbox_callback() {
        return std::bind(&DTPipeline::receive_new_detections, this, std::placeholders::_1, std::placeholders::_2,
                         std::placeholders::_3, std::placeholders::_4);
    }

    /// @brief Serializes boxes into a file
    /// @param begin Iterator to the first box
    /// @param end Iterator to the past-the-end box
    /// @param ts Current timestamp
    /// @param is_valid Boolean that is True if the boxes have been computed
    void receive_new_detections(const EventBbox *begin, const EventBbox *end, timestamp ts, bool is_valid) {
        if (!output_stream_detections_) {
            return;
        }
        assert(output_stream_detections_->good());
        if (!is_valid) {
            assert(begin == nullptr);
            assert(end == nullptr);
            return;
        }
        for (auto box = begin; box != end; ++box) {
            assert(ts == box->t);
            box->write_csv_line(*output_stream_detections_);
        }
    }

    using EventTrackletConsumerCallback =
        std::function<void(const EventTrackedBox *begin, const EventTrackedBox *end, timestamp ts)>;

    /// @brief Returns a callback to handle serialization of tracklets into a file
    /// @return Function to handle created tracks
    EventTrackletConsumerCallback get_track_callback() {
        return std::bind(&DTPipeline::receive_new_tracklets, this, std::placeholders::_1, std::placeholders::_2,
                         std::placeholders::_3);
    }

    /// @brief Serializes tracks into a file
    /// @param begin Iterator to the first track
    /// @param end Iterator to the past-the-end track
    /// @param ts Current timestamp
    void receive_new_tracklets(const EventTrackedBox *begin, const EventTrackedBox *end, timestamp ts) {
        if (!output_stream_tracks_) {
            return;
        }
        assert(output_stream_tracks_->good());
        for (auto track = begin; track != end; ++track) {
            track->write_csv_line(*output_stream_tracks_);
        }
    }

    /// @brief Connects all the components together
    void connect_callbacks() {
        assert(ev_provider_);
        assert(publisher_geom_filter_);
        assert(publisher_noise_filter_);
        assert(object_detector_);
        assert(data_associator_);

        const timestamp start_ts = ev_provider_->get_start_ts();
        publisher_geom_filter_->set_start_ts(start_ts);
        publisher_noise_filter_->set_start_ts(start_ts);
        object_detector_->set_start_ts(start_ts);

        if (display_) {
            display_->set_start_ts(start_ts);
        }

        ev_provider_->set_callback(publisher_geom_filter_->get_event_callback());

        publisher_geom_filter_->add_callbacks(object_detector_->get_event_callback(),
                                              object_detector_->get_timestamp_callback());
        publisher_geom_filter_->add_callbacks(publisher_noise_filter_->get_event_callback(),
                                              publisher_noise_filter_->get_timestamp_callback());

        publisher_noise_filter_->add_callbacks(data_associator_->get_event_callback(),
                                               data_associator_->get_timestamp_callback());

        object_detector_->add_box_consumer_callback(data_associator_->get_box_callback());

        if (output_stream_detections_) {
            object_detector_->add_box_consumer_callback(this->get_bbox_callback());
        }

        if (output_stream_tracks_) {
            data_associator_->add_tracklet_consumer_cb(this->get_track_callback());
        }

        if (display_) {
            publisher_noise_filter_->add_callbacks(display_->get_event_callback(), display_->get_timestamp_callback());
            object_detector_->add_box_consumer_callback(display_->get_box_callback());
            data_associator_->add_tracklet_consumer_cb(display_->get_track_callback());
            display_->set_detector_labels(object_detector_->get_labels());
        }
    }

    /// @brief Starts the camera
    void run() {
        ev_provider_->start();
    }

    /// @brief Stops the pipeline
    void clean_stop() {
        ev_provider_->stop();
        object_detector_->done();
        data_associator_->done();

        if (display_) {
            display_->stop();
        }

        if (output_stream_tracks_) {
            output_stream_tracks_->close();
        }
        if (output_stream_detections_) {
            output_stream_detections_->close();
        }
    }

    std::unique_ptr<EventProviderBase> ev_provider_;
    std::unique_ptr<PreprocessingBase<Event>> geom_preproc_;
    std::unique_ptr<PreprocessingBase<Event>> noise_preproc_;
    std::unique_ptr<Slicer<Event>> publisher_geom_filter_;
    std::unique_ptr<Slicer<Event>> publisher_noise_filter_;
    std::unique_ptr<ObjectDetectorBaseT<Event>> object_detector_;
    std::unique_ptr<DataAssociationBboxTrackedBox<Event>> data_associator_;
    std::unique_ptr<DetectionAndTrackingDisplay<Event>> display_;

    const std::string output_filename_tracks_;
    const std::string output_filename_detections_;

    std::unique_ptr<std::ofstream> output_stream_tracks_;
    std::unique_ptr<std::ofstream> output_stream_detections_;
};

} // namespace Metavision

#endif // METAVISION_SDK_ML_PIPELINE_BUILDER_H
