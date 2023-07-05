/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_OBJECT_DETECTOR_H
#define METAVISION_SDK_ML_OBJECT_DETECTOR_H

#include <functional>
#include <algorithm>
#include <thread>
#include <boost/lockfree/queue.hpp>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/ml/utils/producer_consumer_synchronization.h"
#include "metavision/sdk/ml/algorithms/object_detector_torch_jit.h"

namespace Metavision {

template<typename Event>
class ObjectDetectorBaseT {
public:
    /// @brief Function type handling events
    using EventCallback = std::function<void(const Event *, const Event *)>;

    /// @brief Function to indicate that the time evolves
    using EndSliceCallback = std::function<void(timestamp)>;

    /// @brief Function type handling EventBbox events
    ///
    /// This is the format of the callback of clients of this class
    using EventBoxConsumerCallback =
        std::function<void(const EventBbox *begin, const EventBbox *end, timestamp ts, bool is_valid)>;

    /// @brief Ends the processing of events (no more events)
    virtual void done() = 0;

    /// @brief Registers a new client callback
    /// @param new_box_consumer_callback Function to be called on box generation
    virtual void add_box_consumer_callback(EventBoxConsumerCallback new_box_consumer_callback) = 0;

    /// @brief Returns function to be called on received events to ease lambda function creation
    /// @return Function of type EventCallback
    virtual EventCallback get_event_callback() = 0;

    /// @brief Returns function to be call time to time to update output
    /// @note Every event should be received for this timestamp
    /// @return Function of type EndSliceCallback to update the current timestamp
    virtual EndSliceCallback get_timestamp_callback() = 0;

    /// @brief Gets the labels from the model
    /// @return Vector of string with the label names
    virtual const std::vector<std::string> &get_labels() const = 0;

    /// @brief Gets object detector's accumulation time
    /// @return Accumulation time between two frames
    virtual timestamp get_accumulation_time() const = 0;

    /// @brief Initializes the internal timestamp of the object detector
    ///
    /// This is needed in order to use the start_ts parameter in the pipeline to start at a ts > 0
    ///
    /// @param ts Timestamp of the first considered event
    /// @note Events are not discarded
    virtual void set_start_ts(timestamp ts) = 0;

    /// @brief Updates current detection threshold instead of the default value read from the JSON file
    ///
    /// This is the lower bound on the confidence score for a detection box to be accepted.
    /// It takes values in range ]0;1[
    /// Low value  -> more detections
    /// High value -> less detections
    ///
    /// @param threshold Lower bound on the detection confidence score
    virtual void set_detection_threshold(float threshold) = 0;

    /// @brief Updates current IOU threshold for NMS instead of the default value read from the JSON file
    ///
    /// Non-Maximum suppression discards detection boxes which are too similar to each other, keeping only
    /// the best one of such group. This similarity criterion on based on the measure of Intersection-Over-Union
    /// between the considered boxes.
    /// This threshold is the upper bound on the IOU for two boxes to be considered distinct (and therefore
    /// not filtered out by the Non-Maximum Suppression). It takes values in range ]0;1[
    /// Low value  -> less overlapping boxes
    /// High value -> more overlapping boxes
    ///
    /// @param threshold Upper bound on the IOU for two boxes to be considered distinct
    virtual void set_iou_threshold(float threshold) = 0;
};

/// @brief Generates from events boxes based on a machine learning kernel
///
/// The box generation happens in several steps:
///     - Generates frame from events
///     - Runs detection kernel based on machine learning algorithm
///     - Extracts the detection into boxes vector
///
/// In every case a vector of boxes is generated at every end of event slice.
/// The kernel may be called at a lower frequency than the slice of events
template<typename Event>
class ObjectDetectorT : public ObjectDetectorBaseT<Event> {
public:
    /// @brief Function type handling events
    using EventCallback = std::function<void(const Event *, const Event *)>;

    /// @brief Function to indicate that the time evolves
    using EndSliceCallback = std::function<void(timestamp)>;

    /// @brief Function type handling EventBbox events
    ///
    /// This is the format of the callback of clients of this class
    using EventBoxConsumerCallback =
        std::function<void(const EventBbox *begin, const EventBbox *end, timestamp ts, bool is_valid)>;

    /// @brief Creates a object detector component
    ///
    /// @param directory Folder containing the machine learning model
    /// @param runtime Targeted processor supported: cpu, gpu, gpu:[0-9]
    /// @param events_input_width Sensor's width
    /// @param events_input_height Sensor's height
    /// @param network_input_width Network input frame's width
    /// @param network_input_height Network input frame's height
    ObjectDetectorT(const std::string &directory, const std::string &runtime, int events_input_width,
                    int events_input_height, int network_input_width, int network_input_height) :
        object_detector_algo_(directory, events_input_width, events_input_height, network_input_width,
                              network_input_height),
        cd_processor_(object_detector_algo_.get_cd_processor()) {
        assert((runtime.length() == 3) || (runtime.length() == 5));
        bool gpu_available = false;
        if (runtime == "gpu") {
            gpu_available = object_detector_algo_.use_gpu_if_available();
        } else if (runtime.length() == 5) {
            assert(runtime.substr(0, 4) == "gpu:");
            int gpu_id = runtime.at(4) - '0';
            assert((gpu_id >= 0) && (gpu_id <= 9));
            gpu_available = object_detector_algo_.use_gpu_if_available(gpu_id);
        }

        if ((runtime == "gpu" || runtime.length() == 5) && !gpu_available) {
            MV_SDK_LOG_WARNING() << "GPU not available! Trying to run on CPU.";
            // half precision suported only in gpu 
            if (object_detector_algo_.is_half()) {
                throw std::runtime_error("Error: half precision supported only in gpu ");
            }
        } 

        input_frame_wip_.resize(cd_processor_.get_frame_size());
        std::fill(input_frame_wip_.begin(), input_frame_wip_.end(), 0.f);

        input_frame_ready_.resize(cd_processor_.get_frame_size());
        std::fill(input_frame_ready_.begin(), input_frame_ready_.end(), 0.f);

        done_                      = false;
        prod_cons_.data_ready_     = false;
        prod_cons_.data_processed_ = true;

        tasks_.reset(new boost::lockfree::queue<Task>(10));

        thread_detection_ = std::thread(&ObjectDetectorT::run_detection, this);
    }

    /// @brief Ends the processing of events (no more events)
    virtual void done() override final {
        done_ = true;
        prod_cons_.notify();
        if (thread_detection_.joinable()) {
            thread_detection_.join();
        }
    }

    virtual ~ObjectDetectorT() {
        done();
    }

    /// @brief Registers an additional new client callback
    /// @param new_box_consumer_callback function to be called on box generation
    virtual void add_box_consumer_callback(EventBoxConsumerCallback new_box_consumer_callback) {
        if (new_box_consumer_callback) {
            box_consumer_callbacks_.push_back(new_box_consumer_callback);
        }
    }

    /// @brief Returns function to be called on received events to ease lambda function creation
    /// @return Function of type EventCallback to insert new events
    virtual EventCallback get_event_callback() override final {
        return std::bind(&ObjectDetectorT::receive_new_events_cb, this, std::placeholders::_1, std::placeholders::_2);
    }

    /// @brief Returns function to be called time to time to update output
    /// @note Every event should be received for this timestamp
    /// @return Function of type EndSliceCallback to provide time modification
    virtual EndSliceCallback get_timestamp_callback() override final {
        return std::bind(&ObjectDetectorT::receive_end_events_cb, this, std::placeholders::_1);
    }

    ///  @brief Gets the labels from the model
    virtual const std::vector<std::string> &get_labels() const override final {
        return object_detector_algo_.get_labels();
    }

    /// @brief Gets object detector's accumulation time
    virtual timestamp get_accumulation_time() const override final {
        return object_detector_algo_.get_accumulation_time();
    }

    /// @brief Initializes the internal timestamp of the object detector
    ///
    /// This is needed in order to use the start_ts parameter in the pipeline to start at a ts > 0
    ///
    /// @param ts time at which the first slice of time starts
    virtual void set_start_ts(timestamp ts) override final {
        object_detector_algo_.set_ts(ts);
        last_received_ts_   = ts;
        wip_frame_start_ts_ = ts;
    }

    /// @brief Uses this detection threshold instead of the default value read from the JSON file
    ///
    /// This is the lower bound on the confidence score for a detection box to be accepted.
    /// It takes values in range ]0;1[
    /// Low value  -> more detections
    /// High value -> less detections
    ///
    /// @param threshold Lower bound on the confidence score for the detection box to be considered
    virtual void set_detection_threshold(float threshold) override final {
        object_detector_algo_.set_detection_threshold(threshold);
    }

    /// @brief Use this IOU threshold for NMS instead of the default value read from the JSON file
    ///
    /// Non-Maximum suppression discards detection boxes which are too similar to each other, keeping only
    /// the best one of such group. This similarity criterion on based on the measure of Intersection-Over-Union
    /// between the considered boxes.
    /// This threshold is the upper bound on the IOU for two boxes to be considered distinct (and therefore
    /// not filtered out by the Non-Maximum Suppression). It takes values in range ]0;1[
    /// Low value  -> less overlapping boxes
    /// High value -> more overlapping boxes
    ///
    /// @param threshold Lower bound on the IOU for two boxes to be considered identical
    virtual void set_iou_threshold(float threshold) override final {
        object_detector_algo_.set_iou_threshold(threshold);
    }

private:
    /// @brief Update the current input frame of the object detector given the new batch of events
    ///
    /// This callback should be passed to the events provider
    ///
    /// @param begin Pointer on the first event
    /// @param end Pointer on the last event
    inline void receive_new_events_cb(const Event *begin, const Event *end) {
        assert(!done_);
        assert(cd_processor_.get_frame_size() == input_frame_wip_.size());
        cd_processor_(wip_frame_start_ts_, begin, end, input_frame_wip_.data(), (int)input_frame_wip_.size());
    }

    /// @brief Computes the detection and sends results to clients
    void run_detection() {
        while (!done_ || !tasks_->empty()) {
            Task task;
            if (tasks_->pop(task)) {
                if (task.has_data_) {
                    std::vector<EventBbox> detections;
                    prod_cons_.consumer_wait(done_, [&]() {
                        object_detector_algo_.process(input_frame_ready_, std::back_inserter(detections), task.ts_);
                        std::fill(input_frame_ready_.begin(), input_frame_ready_.end(), 0.f);
                    });

                    // send results to clients
                    for (auto &callback : box_consumer_callbacks_) {
                        if (detections.empty()) {
                            callback(nullptr, nullptr, task.ts_, true);
                        } else {
                            callback(&detections[0], &detections[0] + detections.size(), task.ts_, true);
                        }
                    }
                } else {
                    for (auto &callback : box_consumer_callbacks_) {
                        callback(nullptr, nullptr, task.ts_, false);
                    }
                }
            }
        }
    }

    /// @brief Computes the boxes and update clients
    ///
    /// If the timestamp has not reached the expected value (we have not yet received enough events to
    /// compute entirely the input frame), an empty result is returned to the clients
    /// Otherwise, the object detector computes the boxes and returned to the clients
    ///
    /// This callback should be passed to the events provider
    ///
    /// @param ts Timestamp of the finished slice of event
    ///
    /// @note This class is meant to be used behind a slicer thus this function is called
    /// exactly once by the end of the slice and at the right timestamp
    void receive_end_events_cb(timestamp ts) {
        assert(ts != 0);
        assert(!done_);
        if (delta_t_between_two_end_events_ == 0) {
            delta_t_between_two_end_events_ = ts - last_received_ts_;
            assert(delta_t_between_two_end_events_ != 0);
        } else {
            assert(ts == last_received_ts_ + delta_t_between_two_end_events_);
        }

        {
            // input frame is ready to be computed: process it then reset it
            if (ts % object_detector_algo_.get_accumulation_time() == 0) {
                prod_cons_.producer_wait(done_);
                std::swap(input_frame_wip_, input_frame_ready_);
                wip_frame_start_ts_ = ts;
                Task task           = {ts, true};
                while (!tasks_->push(task)) {}
                prod_cons_.data_is_ready();
                prod_cons_.notify();
            } else {
                Task task = {ts, false};
                while (!tasks_->push(task)) {}
            }
        }
        last_received_ts_ = ts;
    }

    /// Structure to define the work that has to be done
    struct Task {
        timestamp ts_;  ///< timestamp of the slice
        bool has_data_; ///< if true during this slice the inference should be run
    };

    /// Synchronization between the event reception and the internal thread
    ProducerConsumerSynchronization prod_cons_;
    std::thread thread_detection_; ///< internal thread to do the inference

    std::atomic<bool> done_; ///< true once the end is reached

    timestamp last_received_ts_               = 0; ///< last received timestamp to do checks
    timestamp delta_t_between_two_end_events_ = 0; ///< time between two timestamp
    timestamp wip_frame_start_ts_             = 0; ///< start timestamp of the current frame being computed

    /// queue of task to share work with the internal thread
    std::unique_ptr<boost::lockfree::queue<Task>> tasks_;
    Frame_t input_frame_wip_;   ///< input frame for object detector (not ready yet: computed incrementally)
    Frame_t input_frame_ready_; ///< input frame for object detector (ready)

    /// Object detector running machine learning kernel
    Metavision::ObjectDetectorTorchJit object_detector_algo_;
    /// Generator of frame from set of events
    CDProcessing &cd_processor_;

    std::vector<EventBoxConsumerCallback> box_consumer_callbacks_; ///< clients of this object
};

} // namespace Metavision
#endif // METAVISION_SDK_ML_OBJECT_DETECTOR_H
