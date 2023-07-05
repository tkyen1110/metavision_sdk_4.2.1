/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_DISPLAY_H
#define METAVISION_SDK_ML_DISPLAY_H

#include <thread>
#include <boost/circular_buffer.hpp>
#include <boost/lockfree/spsc_queue.hpp>

#include "metavision/sdk/base/utils/get_time.h"
#include "metavision/sdk/core/utils/random_color_map.h"
#include "metavision/sdk/core/utils/simple_displayer.h"
#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/events/event_bbox.h"

namespace {

/// @brief Helper to generate the display frame
class MultipleObjectTrackingDisplayer {
public:
    /// @brief Generates the display for Multiple Objects Tracking
    /// @param first Iterator on first track to display
    /// @param last Iterator after last track to display
    /// @param display_mat Existing image frame in which we want to display the tracked objects
    template<typename InputIt>
    void generate_display(InputIt first, InputIt last, cv::Mat &display_mat) {
        unsigned int n_colors = N_COLORS;
        // initialization of ids_manager by setting it to the size of n_colors
        if (ids_manager_.empty()) {
            ids_manager_.resize(n_colors, -1ll);
        }
        // initialization of history_points_for_display with n_colors circular buffers of 10 Point2i
        if (history_points_for_display_.size() == 0) {
            for (unsigned int i = 0; i < n_colors; i++) {
                history_points_for_display_.emplace_back(50);
            }
        }

        for (; first != last; ++first) {
            cv::rectangle(display_mat, cv::Point2i(first->x, first->y),
                          cv::Point2i(first->x + first->w, first->y + first->h), COLORS[first->track_id % N_COLORS], 3);

            const auto color = first->track_id % n_colors;

            // handles when color loops over n_colors
            if (ids_manager_[color] != first->track_id) {
                history_points_for_display_[color].clear();
                ids_manager_[color] = first->track_id;
            }

            history_points_for_display_[color].push_back(cv::Point2i(first->x + first->w / 2, first->y + first->h / 2));

            for (auto point = history_points_for_display_[color].begin();
                 point != history_points_for_display_[color].end(); point++) {
                const auto next_point = std::next(point);
                if (next_point != history_points_for_display_[color].end()) {
                    cv::line(display_mat, *point, *next_point, COLORS[first->track_id % N_COLORS], 2);
                }
            }
        }
    }

private:
    std::vector<boost::circular_buffer<cv::Point2i>> history_points_for_display_;
    std::vector<long long> ids_manager_;
};

/// @brief Displays the detections and the tracks
class DetectionsDisplayer {
public:
    DetectionsDisplayer() {}
    DetectionsDisplayer(const std::vector<std::string> &labels) : labels_(labels) {}

    void set_detector_labels(const std::vector<std::string> &labels) {
        labels_ = labels;
    }

    /// @brief Generates the display for detections
    /// @param first Iterator on first detection to display
    /// @param last Iterator after last detection to display
    /// @param display_mat Existing image frame in which we want to display the tracked objects
    template<typename InputIt>
    inline void generate_display(InputIt first, InputIt last, cv::Mat &display_mat) {
        const auto width_minus_1  = display_mat.cols - 1.f;
        const auto height_minus_1 = display_mat.rows - 1.f;

        for (; first != last; ++first) {
            const auto ebb = *first;

            const cv::Scalar color = CV_RGB(0., 255., 0.);
            const int thickness    = 3;
            const double txt_scale = 1.;

            const auto x1 = std::min(std::max(ebb.x, 0.f), width_minus_1);
            const auto y1 = std::min(std::max(ebb.y, 0.f), height_minus_1);
            const auto x2 = std::min(std::max(ebb.x + ebb.w, 0.f), width_minus_1);
            const auto y2 = std::min(std::max(ebb.y + ebb.h, 0.f), height_minus_1);

            cv::rectangle(display_mat, cv::Point2f(x1, y1), cv::Point2f(x2, y2), color, thickness);

            // display ID, class and confidence
            {
                std::stringstream ss;
                ss << "Class '" << get_label_from_class_id(labels_, ebb.class_id) << "', confidence "
                   << ebb.class_confidence << ", ts " << ebb.t / 1000000.;

                cv::putText(display_mat, ss.str(), cv::Point2f(x1, y1 - 5), cv::FONT_HERSHEY_PLAIN, txt_scale, color);
            }
        }
    }

private:
    /// @brief Returns the name of the class_id
    /// @param labels Vector of label names
    /// @param class_id Class identifier
    /// @return Name of the class associated to the provided identifier
    std::string get_label_from_class_id(const std::vector<std::string> &labels, const int class_id) {
        assert(class_id > 0);
        if (!labels.empty()) {
            return labels.at(class_id);
        } else {
            return std::string("no_label_provided");
        }
    }

    std::vector<std::string> labels_;
};

} // namespace

namespace Metavision {

/// @brief Component which generates the display of detections and tracks
template<typename Event>
class DetectionAndTrackingDisplay {
public:
    /// @brief Constructs a frame builder component for the detection and tracking pipeline
    /// @param width Sensor's width
    /// @param height Sensor's height
    /// @param pipeline_delta_t Temporal period of the pipeline
    /// @param fps Number of frames per seconds
    /// @param output_video_filename Filename of the output video
    /// @param display_window Boolean to display the frame (default is true)
    DetectionAndTrackingDisplay(int width, int height, timestamp pipeline_delta_t, int fps,
                                const std::string output_video_filename = "", bool display_window = true) :
        width_(width),
        height_(height),
        max_step_between_frames_(std::max(1, static_cast<int>((1e6 / fps) / pipeline_delta_t))),
        max_nb_elem_in_queue_(max_delay_in_seconds_ * (1e6 / pipeline_delta_t)),
        done_(false),
        pipeline_delta_t_(pipeline_delta_t) {
        img_queue_.reset(new boost::lockfree::spsc_queue<cv::Mat>(max_nb_elem_in_queue_));
        events_ts_queue_.reset(new boost::lockfree::spsc_queue<timestamp>(max_nb_elem_in_queue_));
        boxes_ts_queue_.reset(new boost::lockfree::spsc_queue<timestamp>(max_nb_elem_in_queue_));
        tracks_ts_queue_.reset(new boost::lockfree::spsc_queue<timestamp>(max_nb_elem_in_queue_));

        // The display does not need to run faster than 25 frame per seconds
        displayer_.reset(new Metavision::SimpleDisplayer("Detection and Tracking", 25));

        if (output_video_filename != "") {
            video_writer_.reset(new cv::VideoWriter(output_video_filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                                    fps, cv::Size(2 * width_, height_)));
        }

        if (display_window) {
            thread_display_window_ = std::thread(&Metavision::SimpleDisplayer::run, displayer_.get());
        }
        thread_build_frame_ = std::thread(&DetectionAndTrackingDisplay::build_frame, this);
        wip_frame_.create(height_, width_, CV_8UC1);
        wip_frame_.setTo(0);
    }

    using KeyBindCallback = std::function<void(int key)>;
    /// @brief Sets callback to handle keyboard events
    /// @param keybind_func Function to be called when a key is pressed
    void set_ui_keys(KeyBindCallback keybind_func) {
        displayer_->set_on_key_pressed_cb(keybind_func);
    }

    ~DetectionAndTrackingDisplay() {
        stop();
    }

    void stop() {
        done_ = true;
        displayer_->stop();
        if (thread_display_window_.joinable()) {
            thread_display_window_.join();
        }
        if (thread_build_frame_.joinable()) {
            thread_build_frame_.join();
        }
    }

    void set_start_ts(timestamp ts) {
        last_box_received_   = ts;
        last_track_received_ = ts;
    }

    /// @brief Sets names of the class detection
    /// @param labels Array of names per class identifier
    void set_detector_labels(const std::vector<std::string> &labels) {
        detections_displayer_.set_detector_labels(labels);
    }

    // callback for receiving events
    using EventCallback = std::function<void(const Event *, const Event *)>;
    /// @brief Returns a function to generate the display from the events
    /// @return Function to generate a display from the events
    EventCallback get_event_callback() {
        return std::bind(&DetectionAndTrackingDisplay::receive_new_events_cb, this, std::placeholders::_1,
                         std::placeholders::_2);
    }

    // callback for receiving end timestamps
    using EndEventCallback = std::function<void(timestamp)>;
    //
    /// @brief Returns callback to be called at the pipeline frequency
    /// @return Function to be called when time progresses
    EndEventCallback get_timestamp_callback() {
        return std::bind(&DetectionAndTrackingDisplay::receive_end_events_cb, this, std::placeholders::_1);
    }

    // callback for receiving detection boxes
    using EventBoxConsumerCallback =
        std::function<void(const EventBbox *begin, const EventBbox *end, timestamp ts, bool is_valid)>;
    //
    /// @brief Returns function to display generated boxes
    /// @return Function to be called on boxes
    EventBoxConsumerCallback get_box_callback() {
        return std::bind(&DetectionAndTrackingDisplay::receive_new_detections_cb, this, std::placeholders::_1,
                         std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
    }

    // @brief callback for receiving tracklets
    using EventTrackletConsumerCallback =
        std::function<void(const EventTrackedBox *begin, const EventTrackedBox *end, timestamp ts)>;

    /// @brief Returns function to display generated tracks
    /// @return Function to display tracks
    EventTrackletConsumerCallback get_track_callback() {
        return std::bind(&DetectionAndTrackingDisplay::receive_new_tracklets_cb, this, std::placeholders::_1,
                         std::placeholders::_2, std::placeholders::_3);
    }

private:
    /// @brief Internal main loop generating the display from received tracks boxes and events
    void build_frame() {
        // Build the frame to have it in memory
        {
            img_to_display_.create(height_, 2 * width_, CV_8UC3);
            auto sub_img_left  = img_to_display_(cv::Rect(0, 0, width_, height_));
            auto sub_img_right = img_to_display_(cv::Rect(width_, 0, width_, height_));
            sub_img_left.setTo(color_bg_);
        }

        // LUT for the colors
        cv::Vec3b colors[3] = {color_bg_, color_on_, color_off_};

        while (!done_) {
            // the function 'create' generates a new matrix only if required
            img_to_display_.create(height_, 2 * width_, CV_8UC3);
            auto sub_img_left  = img_to_display_(cv::Rect(0, 0, width_, height_));
            auto sub_img_right = img_to_display_(cv::Rect(width_, 0, width_, height_));
            timestamp ts       = 0;
            {
                while (!events_ts_queue_->pop(ts) && !done_) {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            }
            if (done_ && (ts == 0))
                break;

            {
                cv::Mat img;
                {
                    while (!img_queue_->pop(img)) {
                        std::this_thread::yield();
                    }
                }
                // In this part the image of event is generated from the mono channel image of events
                // colors is a LUT to translate the mono channel image to a RGB image
                // The generation is done only on the left side
                int nb_col              = img.cols;
                auto img_to_display_ptr = img_to_display_.ptr<cv::Vec3b>(0);
                auto img_ptr            = img.ptr<uint8_t>(0);
                for (int i = 0; i < img_to_display_.rows; ++i) {
                    auto sub_img_row = i * img_to_display_.cols;
                    auto img_row     = i * img.cols;
                    for (int j = 0; j < nb_col; ++j) {
                        img_to_display_ptr[sub_img_row + j] = colors[img_ptr[img_row + j]];
                    }
                }
                std::stringstream msg;
                msg << timestamp_to_utc_string(ts, 0, false);
                cv::putText(sub_img_left, msg.str(), cv::Point(10, 50), cv::FONT_HERSHEY_TRIPLEX, 1.,
                            cv::Vec3b(0, 0, 255));

                {
                    // The left side of the image is copied on the right as the following opencv code would do:
                    //
                    // auto sub_img_left  = img_to_display_(cv::Rect(0, 0, width_, height_));
                    // auto sub_img_right = img_to_display_(cv::Rect(width_, 0, width_, height_));
                    // sub_img_left.setTo(color_bg_);
                    //
                    // however opencv parallelizes the code automatically which slows down the execution
                    int matrix_size         = img_to_display_.rows * img_to_display_.cols;
                    int nb_col              = img_to_display_.cols;
                    int half_nb_col         = nb_col / 2;
                    auto img_to_display_ptr = img_to_display_.ptr<cv::Vec3b>(0);
                    for (int i = 0; i < matrix_size; i += nb_col) {
                        int last = i + half_nb_col;
                        for (int j = i; j < last; ++j) {
                            img_to_display_ptr[j + half_nb_col] = img_to_display_ptr[j];
                        }
                    }
                }
                cv::line(img_to_display_, cv::Point(width_, 0), cv::Point(width_, height_), cv::Scalar(0, 0, 255));
            }

            timestamp ts_boxes;
            timestamp ts_tracks;
            {
                std::vector<EventBbox> current_boxes;
                {
                    while (!boxes_ts_queue_->pop(ts_boxes)) {
                        std::this_thread::yield();
                    }
                    {
                        std::lock_guard<std::mutex> lg(lock_boxes_queue_);
                        assert(!boxes_queue_.empty());
                        current_boxes = std::move(boxes_queue_.front());
                        boxes_queue_.pop();
                    }
                }

                detections_displayer_.generate_display(current_boxes.cbegin(), current_boxes.cend(), sub_img_left);
            }

            {
                std::vector<EventTrackedBox> current_tracks;

                {
                    while (!tracks_ts_queue_->pop(ts_tracks)) {
                        std::this_thread::yield();
                    }

                    assert(ts_boxes == ts_tracks);
                    {
                        std::lock_guard<std::mutex> lg(lock_tracks_queue_);
                        assert(!tracks_queue_.empty());
                        current_tracks = std::move(tracks_queue_.front());
                        tracks_queue_.pop();
                    }
                }
                mot_displayer_.generate_display(current_tracks.begin(), current_tracks.end(), sub_img_right);
            }

            if (video_writer_) {
                video_writer_->write(img_to_display_);
            }

            { displayer_->swap_frame(img_to_display_); }
        }
    }

    /// @brief Handles events reception to generate frame for displaying
    /// @param begin Iterator on first event
    /// @param end Iterator on last event
    void receive_new_events_cb(const Event *begin, const Event *end) {
        // Check if the frame should be generated
        if ((image_index % max_step_between_frames_) != max_step_between_frames_ - 1) {
            return;
        }

        auto *ptr = wip_frame_.ptr<uint8_t>();
        for (auto ev = begin; ev != end; ++ev) {
            ptr[ev->y * width_ + ev->x] = ev->p ? 2 : 1;
        }
    }

    /// @brief Handles new timestamp
    /// @param ts Current timestamp
    void receive_end_events_cb(timestamp ts) {
        // Check if the frame should be generated
        image_index++;
        if ((image_index % max_step_between_frames_) != 0) {
            return;
        }

        bool alert = true;
        while (!events_ts_queue_->push(ts)) {
            if (alert) {
                MV_SDK_LOG_ERROR() << "DetectionAndTrackingDisplay::receive_end_events_cb(): events_ts_queue_ full. "
                                      "Retrying...";
                alert = false;
            }
        };

        alert              = true;
        auto current_frame = wip_frame_.clone();
        while (!img_queue_->push(current_frame)) {
            if (alert) {
                MV_SDK_LOG_ERROR() << "DEBUG FM: receive_end_events_cb(): img_queue_ full. Push failed... Retrying...";
                alert = false;
            }
        }
        wip_frame_.setTo(0);
    }

    /// @brief Receipts new boxes and synchronizes with the internal thread
    /// @param begin Iterator on first box
    /// @param end Iterator on last box
    /// @param ts Current timestamp
    /// @param is_valid Boolean that is True if the detection have run
    void receive_new_detections_cb(const EventBbox *begin, const EventBbox *end, timestamp ts, bool is_valid) {
        assert(ts == last_box_received_ + pipeline_delta_t_);
        last_box_received_ = ts;
        // Check if the frame should be generated
        box_index++;
        if ((box_index % max_step_between_frames_) != 0) {
            return;
        }

        std::vector<EventBbox> detections;
        detections.reserve(std::distance(begin, end));
        std::copy(begin, end, std::back_inserter(detections));

        {
            std::lock_guard<std::mutex> lg(lock_boxes_queue_);
            boxes_queue_.push(std::move(detections));
            assert(boxes_queue_.size() <= max_nb_elem_in_queue_ + 1);
        }
        bool alert = true;
        while (!boxes_ts_queue_->push(ts)) {
            if (alert) {
                MV_SDK_LOG_ERROR()
                    << "DetectionAndTrackingDisplay::receive_end_events_cb(): boxes_ts_queue_ full. Retrying...";
                alert = false;
            }
        }
    }

    /// @brief Receives new tracks and synchronizes with the internal thread
    /// @param begin Iterator on first track
    /// @param end Iterator on last track
    /// @param ts Current timestamp
    void receive_new_tracklets_cb(const EventTrackedBox *begin, const EventTrackedBox *end, timestamp ts) {
        assert(ts == last_track_received_ + pipeline_delta_t_);
        last_track_received_ = ts;

        // Check if the frame should be generated
        track_index++;
        if ((track_index % max_step_between_frames_) != 0) {
            return;
        }

        std::vector<EventTrackedBox> tracks;
        tracks.reserve(std::distance(begin, end));
        std::copy(begin, end, std::back_inserter(tracks));

        {
            std::lock_guard<std::mutex> lg(lock_tracks_queue_);
            tracks_queue_.push(std::move(tracks));
            assert(tracks_queue_.size() <= max_nb_elem_in_queue_ + 1);
        }
        bool alert = true;
        while (!tracks_ts_queue_->push(ts)) {
            if (alert) {
                MV_SDK_LOG_ERROR() << "DetectionAndTrackingDisplay::receive_new_tracklets_cb(): tracks_ts_queue_ full. "
                                      "Retrying...";
                alert = false;
            }
        }
    }

    // dimensions of the input frame
    const int width_;
    const int height_;

    // amount of lag before we display a warning message
    // and start to block the execution of the pipeline
    const float max_delay_in_seconds_ = .2f;
    // Number of steps between two frame generation
    const int max_step_between_frames_;
    const unsigned int max_nb_elem_in_queue_;

    bool done_ = false;

    const timestamp pipeline_delta_t_;

    // events
    std::unique_ptr<boost::lockfree::spsc_queue<cv::Mat>> img_queue_;
    std::unique_ptr<boost::lockfree::spsc_queue<timestamp>> events_ts_queue_;
    int image_index = 0;

    // bounding boxes
    std::mutex lock_boxes_queue_;
    std::queue<std::vector<EventBbox>> boxes_queue_;
    std::unique_ptr<boost::lockfree::spsc_queue<timestamp>> boxes_ts_queue_;
    int box_index                = 0;
    timestamp last_box_received_ = 0;

    // tracks
    std::mutex lock_tracks_queue_;
    std::queue<std::vector<EventTrackedBox>> tracks_queue_;
    std::unique_ptr<boost::lockfree::spsc_queue<timestamp>> tracks_ts_queue_;
    int track_index                = 0;
    timestamp last_track_received_ = 0;

    std::unique_ptr<Metavision::SimpleDisplayer> displayer_;
    cv::Mat img_to_display_;
    std::unique_ptr<cv::VideoWriter> video_writer_;

    std::thread thread_display_window_; ///< the displayer executes in this thread

    std::thread thread_build_frame_; ///< the frame is built in this thread

    // used to draw detection boxes the output image
    DetectionsDisplayer detections_displayer_;

    // used to draw tracks in the output image
    MultipleObjectTrackingDisplayer mot_displayer_;

    // used to compute the frame to display
    cv::Mat wip_frame_;

    // colors
    cv::Vec3b color_bg_  = cv::Vec3b(52, 37, 30);
    cv::Vec3b color_on_  = cv::Vec3b(236, 223, 216);
    cv::Vec3b color_off_ = cv::Vec3b(201, 126, 64);
};

} // namespace Metavision

#endif // METAVISION_SDK_ML_DISPLAY_H
