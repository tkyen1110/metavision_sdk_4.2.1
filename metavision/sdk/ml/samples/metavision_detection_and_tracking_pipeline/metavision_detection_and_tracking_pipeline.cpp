/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <iostream>
#include <boost/program_options.hpp>

#include "metavision/sdk/base/utils/log.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/ml/pipeline_builder.h"

namespace po = boost::program_options;

using namespace Metavision;

bool parse_arguments(int argc, const char *argv[], DTPipeline_params &params) {
    // clang-format off
    po::options_description pipelineOptions("Pipeline options");
    pipelineOptions.add_options()
        ("help,h", "produce help message")
        ;

    po::options_description inputOptions("Input options");
    inputOptions.add_options()
        ("record-file", po::value<std::string>(&params.record_file_)->default_value(""),
                        "Filename to read events from. Leave empty to use live cam")
        ("start-ts", po::value<timestamp>(&params.start_ts_)->default_value(0), "start timestamp (in µs)")
        ("end-ts", po::value<timestamp>(&params.end_ts_)->default_value(std::numeric_limits<timestamp>::max()),
                   "end timestamp (in µs)")
        ("pipeline-delta-t", po::value<timestamp>(&params.pipeline_delta_t_)->default_value(10000),
                             "data accumulation time for EventsIterator (in µs)")
        ;

    po::options_description outputOptions("Output options");
    outputOptions.add_options()
        ("output-detections-filename", po::value<std::string>(&params.output_detections_filename_)->default_value(""),
                                      "Filename to write detected bbox, in csv format")
        ("output-tracks-filename", po::value<std::string>(&params.output_tracks_filename_)->default_value(""),
                                   "Filename to write tracked bbox, in csv format")
        ("output-video-filename", po::value<std::string>(&params.output_video_filename_)->default_value(""),
                                  "Filename to write a video, in avi format")
        ;

    po::options_description detectorOptions("Object Detector options");
    detectorOptions.add_options()
        ("object-detector-dir", po::value<std::string>(&params.object_detector_dir_),
                                    "Directory of the object detector, including two files named: 1) 'model.ptjit' 2)'info_ssd_jit.json'; "
                                    "INFO: 'model.ptjit' is a Pytorch model compiled and exported with TorchScript; 'info_ssd_jit.json' is a file which contains "
                                    "hyperparameters used to train the model")
        ("detector-confidence-threshold", po::value<float>(&params.detector_confidence_threshold_),
                                          "Use this confidence threshold value instead of the one stored"
                                          " in the model's json. A valid value should be in: ]0., 1[")
        ("detector-NMS-IOU-threshold", po::value<float>(&params.detector_NMS_IOU_threshold_),
                                       "Use this IOU (Intersection Over Union) threshold instead of the one stored"
                                       " in the model's json. A valid value should be in: ]0., 1[")
        ("ml-runtime", po::value<std::string>(&params.ml_runtime_)->default_value("gpu"),
                       "Machine Learning Runtime; Choice of 'cpu', 'gpu', or ('gpu:0', 'gpu:1', etc."
                       " if several gpu are available)")
        ("network-input-width", po::value<int>(&params.network_input_width_)->default_value(0),
                                "Neural Network input width (by default same as event frame width)")
        ("network-input-height", po::value<int>(&params.network_input_height_)->default_value(0),
                                 "Neural Network input height (by default same as event frame height")
        ;


    po::options_description displayOptions("Display Options");
    displayOptions.add_options()
        ("display", po::bool_switch(&params.display_)->default_value(false),
                    "Enable output display")
        ("fps", po::value<int>(&params.fps_)->default_value(20),
                "Number of frames generated per second")
        ;

    po::options_description geometricOptions("Geometric options");
    geometricOptions.add_options()
        ("camera-roi", po::value<std::vector<int>>(&params.camera_roi_)->multitoken(),
                       "Camera ROI [x, y, width, height].")
        ("transpose-input", po::bool_switch(&params.transpose_input_)->default_value(false),
                            "Apply transpose to input. This transformation is applied first."
                            " Note: this transformation changes input dimensions."
                            " Combine with --flipX (resp. --flipY) to rotate 90 deg clockwise"
                            " (resp. counter-clockwise)")
        ("flipX", po::bool_switch(&params.flipX_)->default_value(false),
                  "Flip events horizontally (combine with --transpose-input to rotate 90 deg clockwise,"
                  " or with --flipY to rotate 180 deg)")
        ("flipY", po::bool_switch(&params.flipY_)->default_value(false),
                  "Flip events vertically (combine with --transpose-input to rotate 90 deg counter-clockwise,"
                  " or with --flipX to rotate 180 deg)")
        ;

    po::options_description noiseOptions("Noise Filtering options");
    noiseOptions.add_options()
        ("noise-filtering-type", po::value<std::string>(&params.noise_filtering_type_)->default_value("trail"),
                                 "Type of noise filtering: stc or trail")
        ("noise-filtering-threshold", po::value<timestamp>(&params.noise_filtering_threshold_)->default_value(10000),
                                      "Length of the time window for STC or Trail filtering (in µs)")
        ;


    po::options_description dataAssocOptions("Data Association options");
    dataAssocOptions.add_options()
        ("detection-merge-weight", po::value<float>(&params.detection_merge_weight_)->default_value(0.7f),
                                   "Weight used to compute weighted average of detection and already tracked position "
                                   "(Pos = Detection * Weight + (1 - Weight) * Pos")
        ("deletion-time", po::value<timestamp>(&params.deletion_time_)->default_value(100000),
                          "Time without activity after which the track is detected")
        ("max-iou-inter-track", po::value<float>(&params.max_iou_inter_track_)->default_value(0.5f),
                                "Maximum Intersection Over Union (IOU) inter tracklet before deleting one")
        ("iou-to-match-a-detection", po::value<float>(&params.iou_to_match_a_detection_)->default_value(0.2f),
                                     "Minimum Intersection Over Union (IOU) to match a detection and existing track")
        ("max-iou-for-one-det-to-many-tracks",
         po::value<float>(&params.max_iou_for_one_det_to_many_tracks_)->default_value(0.5f),
         "Threshold at which the tracking is not done if several tracks match with a higher IOU")
        ("use-descriptor", po::bool_switch(&params.use_descriptor_)->default_value(false),
                           "Boolean to enable the use of a Histogram Of Gradient (HOG) descriptor")
        ("number-of-consecutive-detections-to-create-a-new-track",
         po::value<int>(&params.number_of_consecutive_detections_to_create_a_new_track_)->default_value(1),
         "Number of consecutive detections to create a new track")
        ("timesurface-delta-t", po::value<timestamp>(&params.timesurface_delta_t_)->default_value(200000),
                                "Time after which the event are removed from the timesurface")
        ("do-not-update-tracklets-between-detections", po::bool_switch(&params.do_not_update_tracklets_between_detections_)->default_value(false),
                                "Disable update of tracklets between detections (only update when a new detection is received)")
        ;
    // clang-format on

    po::options_description all("Detection and Tracking pipeline");
    all.add(pipelineOptions)
        .add(inputOptions)
        .add(outputOptions)
        .add(displayOptions)
        .add(geometricOptions)
        .add(noiseOptions);
    all.add(detectorOptions).add(dataAssocOptions);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, all), vm);
    po::notify(vm);

    if (params.record_file_ == "") {
        MV_LOG_INFO() << "No recording is specified. Using live camera...\n";
        if (params.start_ts_ != 0) {
            throw std::logic_error("Error: --start-ts is invalid when using a live camera\n");
        }
        if (params.end_ts_ != std::numeric_limits<timestamp>::max()) {
            throw std::logic_error("Error: --end-ts is invalid when using a live camera\n");
        }
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << all;
        return false;
    }

    // Set ERC to 20Mev/s
    params.limit_event_rate_ = true;
    params.max_event_rate_   = 20000000;

    return true;
}

int main(int argc, const char *argv[]) {
    // Parse the command line arguments to build pipeline parameters
    DTPipeline_params params;

    try {
        bool params_are_valid = parse_arguments(argc, argv, params);
        if (!params_are_valid) {
            return 0;
        }
        // Configure a pipeline
        DTPipeline<EventCD> pipeline(params.output_tracks_filename_, params.output_detections_filename_);
        // Runs the pipeline
        pipeline.build_and_run(params);
    } catch (const std::exception &e) { MV_LOG_ERROR() << e.what() << std::endl; }
    return 0;
}
