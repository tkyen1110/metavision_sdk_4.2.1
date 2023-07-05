/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_DATA_ASSOCIATION_H
#define METAVISION_SDK_ML_DATA_ASSOCIATION_H

#include <functional>
#include <algorithm>
#include <atomic>
#include <thread>
#include <queue>
#include <mutex>
#include <boost/lockfree/spsc_queue.hpp>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/core/utils/mostrecent_timestamp_buffer.h"
#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/ml/events/event_tracked_box.h"
#include "metavision/sdk/core/utils/similarity_metrics.h"
#include "metavision/sdk/ml/utils/hog_descriptor.h"

namespace Metavision {

/// @brief Module that matches detections and builds tracklets
template<typename Event, typename DetectionBox, typename Tracklet>
class DataAssociation {
public:
    /// Function type handling events
    using EventCallback = std::function<void(const Event *, const Event *)>;

    using BoxCallback = std::function<void(const DetectionBox *, const DetectionBox *, timestamp ts, bool)>;

    /// Function providing clock ticks
    using EndSliceCallback = std::function<void(timestamp)>;
    using TrackletCallback = std::function<void(const Tracklet *, const Tracklet *, timestamp ts)>;

    /// @brief Creates a DataAssociation object
    /// @param detection_merge_weight Weight to merge a tracklet and a detection. Takes a float value in range [0; 1] (0
    /// means use only tracklet box, 1 means use only detection box)
    /// @param deletion_time Time before deleting a tracklet no longer supported by new detections
    /// @param max_iou_inter_track Maximum IOU inter tracklet before deleting the least recently updated one
    /// @param iou_to_match_a_detection Minimum IOU to match a detection
    /// @param max_iou_for_one_det_to_many_tracks High IOU threshold above which a detection is ignored (skipped) if it
    /// is matched with multiple tracks
    /// @param use_descriptor Boolean to enable the use of a descriptor
    /// @param detection_threshold Number of consecutive detections to create a new track
    /// @param width Sensor's width
    /// @param height Sensor's height
    /// @param time_surface_delta_t Delta time for the timesurface
    /// @param update_tracklets_between_detections boolean to determine if tracklets are updated only when new
    /// detections are received
    DataAssociation(float detection_merge_weight = 0.7f, timestamp deletion_time = 100000,
                    float max_iou_inter_track = 0.5f, float iou_to_match_a_detection = 0.2f,
                    float max_iou_for_one_det_to_many_tracks = 0.5f, bool use_descriptor = false,
                    int detection_threshold = 1, int width = 640, int height = 480,
                    timestamp time_surface_delta_t = 200000, bool update_tracklets_between_detections = true) :
        detection_merge_weight_(detection_merge_weight),
        deletion_time_(deletion_time),
        use_descriptor_(use_descriptor),
        max_iou_inter_track_(max_iou_inter_track),
        iou_to_match_a_detection_(iou_to_match_a_detection),
        max_iou_for_one_det_to_many_tracks_(max_iou_for_one_det_to_many_tracks),
        number_of_consecutive_detections_to_create_a_new_track_(detection_threshold),
        nb_object_classes_(0),
        deletion_time_new_tracklet_(deletion_time),
        time_surface_delta_t_(time_surface_delta_t),
        update_tracklets_between_detections_(update_tracklets_between_detections) {
        boxes_updates_.reset(new boost::lockfree::spsc_queue<timestamp>(10));
        events_updates_.reset(new boost::lockfree::spsc_queue<timestamp>(10));
        if (use_descriptor_)
            descriptor_utils_.reset(
                new HOGDescriptor(cv::Size(64, 32), cv::Size(8, 8), cv::Size(8, 8), cv::Size(8, 8), 9));
        time_surface_.reset(new MostRecentTimestampBuffer(height, width, 2));
        thread_associator_ = std::thread(&DataAssociation::association_thread, this);
    }

    /// @brief Ends the internal thread
    void done() {
        need_end_thread_ = true;
        if (thread_associator_.joinable()) {
            thread_associator_.join();
        }
    }

    /// @brief Destructor
    ~DataAssociation() {
        done();
    }

    /// @brief Callback called on events reception
    /// @param start_ev First event iterator
    /// @param end_ev Last event iterator
    inline void receive_events(const Event *start_ev, const Event *end_ev) {
        assert(!need_end_thread_);
        std::copy(start_ev, end_ev, std::back_inserter(current_events_chunk_));
    }

    /// @brief Callback called on boxes reception
    /// @param start_box First box pointer
    /// @param end_box Last box pointer
    /// @param ts Timestamp of boxes
    /// @param is_valid True if boxes have been computed
    inline void receive_boxes(const DetectionBox *start_box, const DetectionBox *end_box, timestamp ts, bool is_valid) {
        assert(!need_end_thread_);
        // not sure what to do if boxes are in the past?
        std::vector<DetectionBox> box_vector;
        std::copy(start_box, end_box, std::back_inserter(box_vector));
        {
            std::lock_guard<std::mutex> lg(lock_boxes_queue_);
            boxes_queue_.push(std::move(box_vector));
        }

        while (!boxes_updates_->push(ts)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    /// @brief Callback called on clock ticks
    /// @param ts Current timestamp
    inline void receive_end_event_cb(timestamp ts) {
        assert(!need_end_thread_);
        {
            std::lock_guard<std::mutex> lg(lock_events_queue_);
            events_queue_.push(std::move(current_events_chunk_));
        }
        assert(current_events_chunk_.empty());
        current_events_chunk_.reserve(30000);
        while (!events_updates_->push(ts)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    /// @brief Returns the callback to be called on events reception
    /// @return Function of type EventCallback
    EventCallback get_event_callback() {
        return [this](const Event *start_ev, const Event *end_ev) { this->receive_events(start_ev, end_ev); };
    }

    /// @brief Returns the callback called to process boxes
    /// @return Function to receive the boxes
    BoxCallback get_box_callback() {
        return std::bind(&DataAssociation::receive_boxes, this, std::placeholders::_1, std::placeholders::_2,
                         std::placeholders::_3, std::placeholders::_4);
    }

    /// @brief Returns the callback to be called time to time to update output
    /// @return Function of type EndSliceCallback
    /// @warning Every event should be received for this timestamp
    EndSliceCallback get_timestamp_callback() {
        return std::bind(&DataAssociation::receive_end_event_cb, this, std::placeholders::_1);
    }

    /// @brief Sets a callback that is called when tracklets are computed
    /// @param cb The callback to be called
    void add_tracklet_consumer_cb(TrackletCallback cb) {
        tracklets_clients_cb_.push_back(cb);
    }

    /// @brief Disables the update of tracks positions between detections
    void disable_update_tracklets_positions_between_detections() {
        update_tracklets_between_detections_ = false;
    }

private:
    /// @brief Calls clients callbacks
    void send_tracklets_to_clients() {
        std::lock_guard<std::mutex> lg(lock_tracklets_);
        for (auto callback : tracklets_clients_cb_) {
            if (tracklets_.empty()) {
                callback(nullptr, nullptr, last_association_);
            } else {
                std::vector<Tracklet> tracked_boxes;
                for (auto &it : tracklets_) {
                    it.track.t = last_association_;
                    tracked_boxes.push_back(it.track);
                }
                callback(&tracked_boxes[0], &tracked_boxes[0] + tracked_boxes.size(), last_association_);
            }
        }
    }

    /// @brief Internal processing loop
    ///
    /// Pops the boxes and events and runs the data association.
    void association_thread() {
        timestamp boxes_last_update;
        timestamp events_last_update;
        std::vector<Event> event_vec;
        std::vector<DetectionBox> boxes;
        // as long as our last box update is same as the events one, or that we did not get any box, sleep
        while (!events_updates_->empty() || !boxes_updates_->empty() || !need_end_thread_) {
            {
                while (events_updates_->empty() && boxes_updates_->empty() && !need_end_thread_) {
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
                if (events_updates_->empty() && boxes_updates_->empty() && need_end_thread_) {
                    break;
                }
            }

            while (!events_updates_->pop(events_last_update)) {}
            {
                std::lock_guard<std::mutex> lg(lock_events_queue_);
                assert(!events_queue_.empty());
                event_vec = std::move(events_queue_.front());
                events_queue_.pop();
            }
            update_ts(event_vec, events_last_update);
            while (!boxes_updates_->pop(boxes_last_update)) {}
            {
                {
                    std::lock_guard<std::mutex> lg(lock_boxes_queue_);
                    assert(!boxes_queue_.empty());
                    boxes = std::move(boxes_queue_.front());
                    boxes_queue_.pop();
                }
            }
            assert(boxes_last_update == events_last_update);
            // update time surface
            do_data_association(boxes, boxes_last_update);
            send_tracklets_to_clients();
        }
    }

    struct CustomTracks {
        EventTrackedBox track;
        std::vector<float> descriptor;
        DetectionBox detection;
        float dx = 0.f;
        float dy = 0.f;
        float dw = 0.f;
        float dh = 0.f;
    };

    /// @brief Adds a detection to the tracks list
    /// @param detection Box representing an object
    /// @param active_tracks Array of existing tracks
    /// @param ts Current timestamp
    /// @param descriptor Descriptor of the box
    void create_track_from_detection(const DetectionBox &detection, std::vector<CustomTracks> &active_tracks,
                                     timestamp ts, std::vector<float> &descriptor) {
        EventTrackedBox track(ts, detection.x, detection.y, detection.w, detection.h, detection.class_id,
                              next_available_id_, detection.class_confidence);
        active_tracks.push_back({track, descriptor, detection});
        MV_SDK_LOG_DEBUG() << "\n========\ncreated new track with id:" << next_available_id_;

        ++next_available_id_;
    }

    /// @brief Merges a detection with an existing track
    /// @param bbox Box representing an object
    /// @param active_bbox Array of existing tracks
    /// @param ts Current timestamp
    /// @param descriptor Descriptor of the box
    /// @param similarity_box_track Best matching value found
    void merge_single_detection(const DetectionBox &bbox, CustomTracks &active_bbox, timestamp ts,
                                std::vector<float> &descriptor, float similarity_box_track) {
        auto &weight = detection_merge_weight_;
        // Update coordinates
        active_bbox.track.x = (1 - weight) * active_bbox.track.x + weight * bbox.x;
        active_bbox.track.w = (1 - weight) * active_bbox.track.w + weight * bbox.w;
        active_bbox.track.y = (1 - weight) * active_bbox.track.y + weight * bbox.y;
        active_bbox.track.h = (1 - weight) * active_bbox.track.h + weight * bbox.h;
        if (!descriptor.empty()) {
            // Updates the track descriptor with newly detected box's descriptor by computing a weighted median
            auto merge_weighted = [weight](float a, float b) -> float { return weight * a + (1 - weight) * b; };
            std::transform(descriptor.begin(), descriptor.end(), active_bbox.descriptor.begin(),
                           active_bbox.descriptor.begin(), merge_weighted);
        }
        // Update time of update
        active_bbox.track.set_last_detection_update(ts, bbox.class_confidence, similarity_box_track);
        active_bbox.track.nb_detections++;
        // Update last detection box
        float dt_seconds      = (bbox.t - active_bbox.detection.t) * 1e-6f;
        active_bbox.dx        = (bbox.x - active_bbox.detection.x) / dt_seconds;
        active_bbox.dy        = (bbox.y - active_bbox.detection.y) / dt_seconds;
        active_bbox.dw        = (bbox.w - active_bbox.detection.w) / dt_seconds;
        active_bbox.dh        = (bbox.h - active_bbox.detection.h) / dt_seconds;
        active_bbox.detection = bbox;
    }

    /// @brief Tries to match the detection with not associated tracks
    /// @param curr_boxes Array of detected boxes
    /// @param ts Current timestamp
    void match_boxes_to_tracklets(const std::vector<DetectionBox> &curr_boxes, timestamp ts) {
        std::lock_guard<std::mutex> lg(lock_tracklets_);
        // Penalize previous tracking confidence before association
        for (auto it_track = tracklets_.begin(); it_track != tracklets_.end(); ++it_track) {
            it_track->track.tracking_confidence *= tracking_confidence_decay_;
        }
        if (curr_boxes.size() > 0) {
            // Fill map with all current active bboxes and initialize similarity to 0
            std::map<int, float> unassociated_tracks;
            using SizeType = typename std::vector<CustomTracks>::size_type;
            for (SizeType idx_tracklet = 0; idx_tracklet < tracklets_.size(); ++idx_tracklet) {
                unassociated_tracks[idx_tracklet] = 0.f;
            }

            for (auto box : curr_boxes) {
                float best_similarity            = 0.f;
                auto best_candidate              = unassociated_tracks.end();
                size_t amount_of_good_candidates = 0;
                std::vector<float> detection_descriptor;
                if (use_descriptor_) {
                    cv::Rect detection_rect(box.x, box.y, box.w, box.h);
                    std::lock_guard<std::mutex> lg_time_surface(lock_time_surface_);
                    descriptor_utils_->compute_description(time_surface_image_, detection_rect, detection_descriptor);
                }
                for (auto unassociated_it = unassociated_tracks.begin(); unassociated_it != unassociated_tracks.end();
                     ++unassociated_it) {
                    int current_track_idx = unassociated_it->first;
                    auto &track           = tracklets_[current_track_idx];
                    float similarity;
                    if (nb_object_classes_ == 0) {
                        similarity = Metavision::Utils::compute_similarity_iou_using_classid(box, track.track);
                        MV_SDK_LOG_DEBUG()
                            << "DataAssociationAlgorithm::process: using compute_similarity_iou_using_classid "
                               "without similarity matrix:"
                            << similarity;
                    } else {
                        assert(nb_object_classes_ >= 1);
                        assert(similarity_matrix_.size() == (nb_object_classes_ + 1) * (nb_object_classes_ + 1));
                        similarity = Metavision::Utils::compute_similarity_iou_using_classid_and_similarity_matrix(
                            box, track.track, similarity_matrix_, nb_object_classes_);
                        MV_SDK_LOG_DEBUG()
                            << "DataAssociationAlgorithm::process: using compute_similarity_iou_using_classid with "
                               "similarity matrix:"
                            << similarity;
                    }

                    float desc_simi = -1.f;
                    if (use_descriptor_) {
                        desc_simi = descriptor_utils_->compute_similarity(detection_descriptor, track.descriptor);
                    }
                    unassociated_it->second = similarity;
                    // if we pass the minimal test
                    if (similarity > iou_to_match_a_detection_) {
                        // update the amount of CustomTracks matching that detection high enough
                        if (similarity > max_iou_for_one_det_to_many_tracks_) {
                            amount_of_good_candidates++;
                        }
                        // if we have the highest score yet, memorize
                        float weighted_similarity = similarity;
                        if (use_descriptor_) {
                            weighted_similarity *= desc_simi;
                        }
                        if (weighted_similarity > best_similarity) {
                            best_similarity = weighted_similarity;
                            best_candidate  = unassociated_it;
                        }
                    }
                }

                // If we found an association, merge the detection into the active bbox
                if (best_candidate != unassociated_tracks.end()) {
                    if (amount_of_good_candidates < 2) {
                        int track_to_merge_idx = best_candidate->first;
                        auto &track_to_merge   = tracklets_.at(track_to_merge_idx).track;
                        MV_SDK_LOG_DEBUG()
                            << Log::no_space << "Found association between detection : " << box.class_id << " -> ("
                            << box.x << ", " << box.y << ", " << box.w << ", " << box.h << ") "
                            << " and track : " << track_to_merge.track_id << " -> (" << track_to_merge.x << ", "
                            << track_to_merge.y << ", " << track_to_merge.w << ", " << track_to_merge.h
                            << ")    similarity was " << best_similarity;
                        merge_single_detection(box, tracklets_.at(track_to_merge_idx), ts, detection_descriptor,
                                               best_similarity);
                        // Remove from unassociated boxes.
                        unassociated_tracks.erase(best_candidate);
                    } else {
                        MV_SDK_LOG_DEBUG()
                            << "DataAssociationAlgorithm::process: we have found multiple associations for "
                               "detection:"
                            << box;
                    }

                } else {
                    MV_SDK_LOG_DEBUG() << Log::no_space << "Create a new track with detection at ts: " << ts << "  : "
                                       << box.class_id << " -> (" << box.x << ", " << box.y << ", " << box.w << ", "
                                       << box.h << ")";
                    // If no active bbox overlaps with the detection, create a new active bbox
                    create_track_from_detection(box, tracklets_, ts, detection_descriptor);
                }
            } // end for loop for boxes
        }     // end if boxes empty
    }

    /// @brief Updates tracks position even when no detection is received
    /// @param ts Current Timestamp
    void update_tracklets_without_new_detections(timestamp ts) {
        std::lock_guard<std::mutex> lg(lock_tracklets_);
        for (auto it_track = tracklets_.begin(); it_track != tracklets_.end(); ++it_track) {
            if (it_track->detection.t != ts) {
                const float dt = (ts - last_association_) * 1e-6f;
                it_track->track.x += dt * it_track->dx;
                it_track->track.y += dt * it_track->dy;
                it_track->track.w += dt * it_track->dw;
                it_track->track.h += dt * it_track->dh;
            }
        }
    }

    /// @brief Removes too old and overlapping tracklets
    /// @param ts Current Timestamp
    void cleanup_tracklets(timestamp ts) {
        // sort by time of last detection: the first tracks will be the ones with the most recent updates
        std::lock_guard<std::mutex> lg(lock_tracklets_);

        std::sort(tracklets_.begin(), tracklets_.end(), [this](const CustomTracks &a, const CustomTracks &b) {
            if (a.track.nb_detections >= number_of_consecutive_detections_to_create_a_new_track_) {
                if (b.track.nb_detections >= number_of_consecutive_detections_to_create_a_new_track_) {
                    // both tracks are validated, compare last_detection_update_time
                    return b.track.last_detection_update_time < a.track.last_detection_update_time;
                } else {
                    // only a is validated. It should be sorted first regardless of last_detection_update_time
                    return true;
                }
            } else {
                if (b.track.nb_detections >= number_of_consecutive_detections_to_create_a_new_track_) {
                    // only b is validated. It should be sorted first regardless of last_detection_update_time
                    return false;
                } else {
                    // neither track is validated, compare last_detection_update_time
                    return b.track.last_detection_update_time < a.track.last_detection_update_time;
                }
            }
        });

        for (auto track = tracklets_.begin(); track != tracklets_.end();) {
            // delete the current track if it is too old
            if ((ts - track->track.last_detection_update_time) > deletion_time_) {
                MV_SDK_LOG_DEBUG() << "Delete track" << track->track.track_id << "at time" << ts;
                MV_SDK_LOG_DEBUG() << track->track.nb_detections;
                track = tracklets_.erase(track);
                continue;
            }

            // delete the current track if it is recent but has not been confirmed
            if ((track->track.nb_detections < number_of_consecutive_detections_to_create_a_new_track_) &&
                (ts - track->track.last_detection_update_time > deletion_time_new_tracklet_)) {
                MV_SDK_LOG_DEBUG() << "Delete recently created track" << track->track.track_id << "at time" << ts;
                MV_SDK_LOG_DEBUG() << track->track.nb_detections << " vs "
                                   << number_of_consecutive_detections_to_create_a_new_track_;
                track = tracklets_.erase(track);
                continue;
            }

            // delete tracks supported by less recently updated detection if their similarity is too high with current
            // track
            for (auto second_track = std::next(track); second_track != tracklets_.end();) {
                float similarity;
                if (nb_object_classes_ == 0) {
                    similarity =
                        Metavision::Utils::compute_similarity_iou_using_classid(track->track, second_track->track);
                } else {
                    assert(nb_object_classes_ >= 1);
                    assert(similarity_matrix_.size() == (nb_object_classes_ + 1) * (nb_object_classes_ + 1));
                    similarity = Metavision::Utils::compute_similarity_iou_using_classid_and_similarity_matrix(
                        track->track, second_track->track, similarity_matrix_, nb_object_classes_);
                }
                MV_SDK_LOG_DEBUG() << "similarity from track" << track->track.track_id << "to"
                                   << second_track->track.track_id << "is" << similarity;
                if (similarity > max_iou_inter_track_) {
                    MV_SDK_LOG_DEBUG() << "Delete track" << second_track->track.track_id << "based on similarity";
                    second_track = tracklets_.erase(second_track);
                } else {
                    second_track++;
                }
            }
            track++;
        }
    }

    /// @brief Main data association method
    ///
    /// Tries to match the detection boxes and cleans the old tracklets.
    ///
    /// @param curr_boxes Array of detected boxes
    /// @param ts Current timestamp
    void do_data_association(const std::vector<DetectionBox> &curr_boxes, timestamp ts) {
        // this ensures we don't do the data association twice
        match_boxes_to_tracklets(curr_boxes, ts);
        if (update_tracklets_between_detections_) {
            update_tracklets_without_new_detections(ts);
        }
        cleanup_tracklets(ts);
        last_association_ = ts;
    }

    /// @brief Updates the timesurface
    /// @param event_vec Vector of events
    /// @param ts Current timestamp
    void update_ts(const std::vector<Event> &event_vec, timestamp ts) {
        std::lock_guard<std::mutex> lg_time_surface(lock_time_surface_);
        for (auto &ev : event_vec) {
            time_surface_->at(ev.y, ev.x, ev.p) = ev.t;
        }

        time_surface_->generate_img_time_surface(ts, time_surface_delta_t_, time_surface_image_);
    }

    /// @brief Sets the similarity matrix used to track boxes
    /// @param nb_object_classes Number of available classes
    /// @param similarity_matrix Matrix to used to compute track matching
    void use_similarity_matrix(unsigned int nb_object_classes, const std::vector<float> &similarity_matrix) {
        assert(nb_object_classes > 0);
        assert(similarity_matrix.size() == (nb_object_classes + 1) * (nb_object_classes + 1));
        nb_object_classes_ = nb_object_classes;
        similarity_matrix_ = similarity_matrix;

        if (Metavision::LogLevel::Debug >= getLogLevel()) {
            auto log = MV_SDK_LOG_DEBUG()
                       << Log::no_endline << Log::no_space << "DEBUG: similarity matrix used in data_association: ";
            for (auto row = 0; row < nb_object_classes + 1; ++row) {
                for (auto col = 0; col < nb_object_classes + 1; ++col) {
                    log << similarity_matrix_[row * (nb_object_classes + 1) + col] << "   ";
                }
                log << std::endl;
            }
        }
    }

    std::mutex lock_boxes_queue_;
    std::queue<std::vector<DetectionBox>> boxes_queue_;
    std::mutex lock_events_queue_;
    std::queue<std::vector<Event>> events_queue_;
    std::vector<Event> current_events_chunk_;
    std::unique_ptr<boost::lockfree::spsc_queue<timestamp>> boxes_updates_;
    std::unique_ptr<boost::lockfree::spsc_queue<timestamp>> events_updates_;
    timestamp last_association_{0};
    std::atomic<bool> need_end_thread_{false};
    std::thread thread_associator_;
    std::mutex lock_tracklets_;
    std::vector<CustomTracks> tracklets_;

    float detection_merge_weight_;
    uint32_t next_available_id_ = 1;
    timestamp deletion_time_;
    bool use_descriptor_;

    /// Allowed IOU between two tracks
    float max_iou_inter_track_;

    /// Minimum IOU to accept a match with a detection
    float iou_to_match_a_detection_;

    /// IOU threshold above which a detection with one track is not updated (the detection is ignored)
    float max_iou_for_one_det_to_many_tracks_;

    /// In order to be properly created and returned, a new track must be supported by a certain number of
    /// consecutive detections
    int number_of_consecutive_detections_to_create_a_new_track_;

    /// Determine if tracklets should be updated between detections
    bool update_tracklets_between_detections_;

    unsigned int nb_object_classes_;
    std::vector<float> similarity_matrix_;

    std::unique_ptr<HOGDescriptor> descriptor_utils_;
    timestamp deletion_time_new_tracklet_;
    std::mutex lock_time_surface_;
    std::unique_ptr<MostRecentTimestampBuffer> time_surface_;
    cv::Mat time_surface_image_;
    const timestamp time_surface_delta_t_;
    std::vector<TrackletCallback> tracklets_clients_cb_;

    const float tracking_confidence_decay_ = 0.9f;
};

} // namespace Metavision
#endif // DETECTION_AND_TRACKING_COMMON_DATA_ASSOCIATION_H
