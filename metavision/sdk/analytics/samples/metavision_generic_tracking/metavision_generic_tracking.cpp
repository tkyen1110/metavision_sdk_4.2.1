/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// Example of using SDK to track objects.

#include <mutex>
#include <condition_variable>
#include <set>
#include <boost/program_options.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/analytics/configs/tracking_algorithm_config.h>
#include <metavision/sdk/analytics/algorithms/tracking_algorithm.h>
#include <metavision/sdk/analytics/utils/tracking_drawing.h>
#include <metavision/sdk/core/algorithms/on_demand_frame_generation_algorithm.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

#include "simple_video_writer.h"

// Utility function to parse command line
namespace boost_po = boost::program_options;

using TrackingBuffer = std::vector<Metavision::EventTrackingData>;

class Pipeline {
public:
    Pipeline() = default;

    ~Pipeline() = default;

    bool parse_command_line(int argc, char *argv[]);

    /// @brief Initializes the camera
    bool initialize_camera();

    /// @brief Initializes the filters
    void initialize_tracker();

    /// @brief Starts the camera
    bool start();

    /// @brief Stops the pipeline
    void stop();

    /// @brief Waits until the end of the file or until the exit of the display
    void run();

private:
    /// @brief Processing of the output of the @ref Metavision::TrackingAlgorithm.
    void tracker_callback(Metavision::timestamp ts, TrackingBuffer &tracked_objects);

    /// @brief Callback called by the @ref Metavision::Window when a key is pressed.
    void keyboard_callback(Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods);

    /// Tracker's update frequency.
    float update_frequency_;

    /// Min and max size of an object to track.
    int min_size_;
    int max_size_;

    /// Camera's parameters.
    std::string filename_;
    std::string biases_file_ = "";

    /// Display's parameters.
    /// If we display data on the screen or not.
    bool display_;
    /// Filename to save the resulted video.
    std::string output_video_;
    /// Current image in which the result of the tracking is drawn.
    cv::Mat back_img_;

    // Conditional variables to notify the end of the processing.
    std::condition_variable process_cond_;
    std::mutex process_mutex_;
    volatile bool is_processing_ = true;

    /// Time at which to start the video writing.
    /// 0 meaning from the beginning.
    Metavision::timestamp write_from_;
    /// Time at which to stop the video writing.
    /// Max meaning until the end.
    Metavision::timestamp write_to_;

    /// SpatioTemporalContrast threshold (if 0 STC is disable)
    int stc_threshold_;

    std::unique_ptr<Metavision::Camera> camera_;                                     ///< Pointer to the camera class
    std::unique_ptr<Metavision::TrackingAlgorithm> tracker_;                         ///< Instance of the tracker
    std::unique_ptr<Metavision::OnDemandFrameGenerationAlgorithm> frame_generation_; ///< Frame generator
    std::unique_ptr<SimpleVideoWriter> video_writer_;                                ///< Video writer
    std::unique_ptr<Metavision::Window> window_;                                     ///< Display window
    std::unique_ptr<Metavision::SpatioTemporalContrastAlgorithm> stc_algo_;          ///< Noise filter

    /// Ids of all the tracked objects.
    std::set<size_t> tracked_object_ids_;

    /// The tracking algorithm being an asynchronous algorithm, some actions cannot be performed while the tracker's
    /// asynchronous callback is being called. Those actions correspond to actions that trigger other calls to the
    /// asynchronous callback (i.e. the asynchronous callback being called recursively). More specifically here, this
    /// happens when we modify the tracker's update frequency by pressing a key (the keyboard callback is called when
    /// calling window_->show() from inside the tracker's callback). As a consequence, commands associated with UI
    /// events must be saved, and their processing done from outside the tracker's callback.
    using TrackerCommand = std::function<void()>;
    TrackerCommand tracker_cmd_;

    /// Vector of events used to filter events
    std::vector<Metavision::EventCD> event_buffer_;
};

bool Pipeline::initialize_camera() {
    // If the filename is set, then read from the file
    if (filename_ != "") {
        try {
            camera_ = std::make_unique<Metavision::Camera>(
                Metavision::Camera::from_file(filename_, Metavision::FileConfigHints().real_time_playback(false)));
        } catch (Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return false;
        }
        // Otherwise, set the input source to the first available camera
    } else {
        try {
            camera_ = std::make_unique<Metavision::Camera>(Metavision::Camera::from_first_available());

            if (biases_file_ != "") {
                camera_->biases().set_from_file(biases_file_);
            }
        } catch (Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return false;
        }
    }

    // Add camera runtime error callback
    camera_->add_runtime_error_callback([](const Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); });

    return true;
}

bool Pipeline::parse_command_line(int argc, char *argv[]) {
    const std::string short_program_desc("Code sample tracking generic moving objects in a stream of events from an "
                                         "event-based device or recorded data.\n");

    const std::string long_program_desc(
        short_program_desc + "Press 'q' to leave the program.\n"
                             "Press 'a' to increase the minimum size of the object to track.\n"
                             "Press 'b' to decrease the minimum size of the object to track.\n"
                             "Press 'c' to increase the maximum size of the object to track.\n"
                             "Press 'd' to decrease the maximum size of the object to track.\n"
                             "Press 'i' to decrease the tracking update frequency.\n"
                             "Press 'j' to increase the tracking update frequency.\n"
                             "It's recommended to set the 'stc-threshold' option to 10000 in case of a live camera.\n");

    boost_po::options_description options_desc;
    boost_po::options_description base_options("Base options");
    // clang-format off
    base_options.add_options()
        ("help,h", "Produce help message.")
        ("input-file,i",     boost_po::value<std::string>(&filename_), "Path to input file. If not specified, the camera live stream is used.")
        ("biases,b",         boost_po::value<std::string>(&biases_file_), "Path to a biases file. If not specified, the camera will be configured with the default biases")
        ("update-frequency", boost_po::value<float>(&update_frequency_)->default_value(200.), "Tracker's update frequency, in Hz")
        ("min-size",         boost_po::value<int>(&min_size_)->default_value(10), "Minimal size of an object to track, in pixels")
        ("max-size",         boost_po::value<int>(&max_size_)->default_value(300), "Maximal size of an object to track, in pixels")
        ("stc-threshold",    boost_po::value<int>(&stc_threshold_)->default_value(0), "Spatio Temporal Contrast threshold (Disabled if the threshold is equal to 0). It's recommended to use 10000 in case of a live camera.")
        ;
    // clang-format on

    boost_po::options_description outcome_options("Outcome options");
    // clang-format off
    outcome_options.add_options()   
        ("display,d",           boost_po::value(&display_)->default_value(true), "Activate display or not")
        ("output-video-file,o", boost_po::value<std::string>(&output_video_), "Path to an output AVI file to save the resulting video. A frame is generated every time the tracking callback is called.")
        ("write-from",          boost_po::value<Metavision::timestamp>(&write_from_)->default_value(0), "Start time to save video (in us)")
        ("write-to",            boost_po::value<Metavision::timestamp>(&write_to_)->default_value(std::numeric_limits<Metavision::timestamp>::max()), "End time to save video (in us)")
        ;
    // clang-format on

    options_desc.add(base_options).add(outcome_options);

    boost_po::variables_map vm;
    try {
        boost_po::store(boost_po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        boost_po::notify(vm);
    } catch (boost_po::error &e) {
        MV_LOG_ERROR() << short_program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return false;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << short_program_desc;
        MV_LOG_INFO() << options_desc;
        return false;
    }

    MV_LOG_INFO() << long_program_desc;

    return true;
}

/// @brief Starts the camera.
bool Pipeline::start() {
    const bool started = camera_->start();
    if (!started) {
        MV_LOG_ERROR() << "The camera could not be started.";
    }

    return started;
}

/// @brief Stops the camera when finished.
void Pipeline::stop() {
    try {
        camera_->stop();
    } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
}

/// @brief Runs the pipeline.
///
/// Displays the window if enabled, otherwise waits for the tracking thread to complete (i.e. the camera's one).
void Pipeline::run() {
    if (window_) {
        while (!window_->should_close()) {
            // we sleep the main thread a bit to avoid using 100% of a CPU's core
            static constexpr std::int64_t kSleepPeriodMs = 20;

            Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
        }
    } else {
        // Waits until the end of the file or until the user presses 'q'.
        std::unique_lock<std::mutex> lock(process_mutex_);
        process_cond_.wait(lock, [this] { return !is_processing_; });
    }

    MV_LOG_INFO() << "Number of tracked objects:" << *tracked_object_ids_.crbegin();
}

/// [GENERIC_TRACKING_TRACKER_CALLBACK_BEGIN]
void Pipeline::tracker_callback(Metavision::timestamp ts, TrackingBuffer &tracked_objects) {
    for (const auto &obj : tracked_objects)
        tracked_object_ids_.insert(obj.object_id_);

    if (frame_generation_) {
        frame_generation_->generate(ts, back_img_);

        Metavision::draw_tracking_results(ts, tracked_objects.cbegin(), tracked_objects.cend(), back_img_);

        if (video_writer_)
            video_writer_->write_frame(ts, back_img_);

        if (window_)
            window_->show(back_img_);
    }
}
/// [GENERIC_TRACKING_TRACKER_CALLBACK_END]

void Pipeline::keyboard_callback(Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
    static constexpr int kSizeStep = 10;

    if (action == Metavision::UIAction::RELEASE) {
        if (key != Metavision::UIKeyEvent::KEY_I && key != Metavision::UIKeyEvent::KEY_J) {
            switch (key) {
            case Metavision::UIKeyEvent::KEY_Q:
            case Metavision::UIKeyEvent::KEY_ESCAPE: {
                std::lock_guard<std::mutex> lock(process_mutex_);
                is_processing_ = false;
                process_cond_.notify_all();
            }
                window_->set_close_flag();
                break;
            case Metavision::UIKeyEvent::KEY_A:
                if (min_size_ + kSizeStep <= max_size_) {
                    min_size_ += kSizeStep;
                    MV_LOG_INFO() << "Setting min size to" << min_size_;
                    tracker_->set_min_size(min_size_);
                }
                break;
            case Metavision::UIKeyEvent::KEY_B:
                if (min_size_ - kSizeStep >= 0) {
                    min_size_ -= kSizeStep;
                    MV_LOG_INFO() << "Setting min size to" << min_size_;
                    tracker_->set_min_size(min_size_);
                }
                break;
            case Metavision::UIKeyEvent::KEY_C:
                if (max_size_ <= std::numeric_limits<int>::max() - kSizeStep) {
                    max_size_ += kSizeStep;
                    MV_LOG_INFO() << "Setting max size to" << max_size_;
                    tracker_->set_max_size(max_size_);
                }
                break;
            case Metavision::UIKeyEvent::KEY_D:
                if (max_size_ - kSizeStep >= min_size_) {
                    max_size_ -= kSizeStep;
                    MV_LOG_INFO() << "Setting max size to" << max_size_;
                    tracker_->set_max_size(max_size_);
                }
                break;
            default:
                break;
            }
        } else {
            tracker_cmd_ = [this, key]() {
                switch (key) {
                case Metavision::UIKeyEvent::KEY_I:
                    if (update_frequency_ / 2 >= 1.) {
                        update_frequency_ /= 2;
                        MV_LOG_INFO() << "Setting update frequency to" << update_frequency_;
                        tracker_->set_update_frequency(update_frequency_);
                    }
                    break;
                case Metavision::UIKeyEvent::KEY_J:
                    frame_generation_->reset();
                    if (update_frequency_ * 2 <= 1000) {
                        update_frequency_ *= 2;
                        MV_LOG_INFO() << "Setting update frequency to" << update_frequency_;
                        tracker_->set_update_frequency(update_frequency_);
                    }
                    break;
                }
            };
        }
    }
}

void Pipeline::initialize_tracker() {
    const auto &geometry    = camera_->geometry();
    const int sensor_width  = geometry.width();
    const int sensor_height = geometry.height();

    if (stc_threshold_ > 0)
        stc_algo_ =
            std::make_unique<Metavision::SpatioTemporalContrastAlgorithm>(sensor_width, sensor_height, stc_threshold_);

    // Creates filters.
    Metavision::TrackingConfig tracking_config;
    tracker_ = std::make_unique<Metavision::TrackingAlgorithm>(sensor_width, sensor_height, tracking_config);
    tracker_->set_update_frequency(update_frequency_);
    tracker_->set_min_size(min_size_);
    tracker_->set_max_size(max_size_);

    if (!output_video_.empty() || display_) {
        frame_generation_.reset(new Metavision::OnDemandFrameGenerationAlgorithm(sensor_width, sensor_height));
    }

    if (!output_video_.empty()) {
        video_writer_.reset(new SimpleVideoWriter(sensor_width, sensor_height,
                                                  static_cast<int>(1000000.f / update_frequency_), output_video_));
        video_writer_->set_write_range(write_from_, write_to_);
    }

    if (display_) {
        window_ = std::make_unique<Metavision::Window>("Metavision Tracking sample", sensor_width, sensor_height,
                                                       Metavision::BaseWindow::RenderMode::BGR);
        // Notifies the pipeline when the window is exited.
        window_->set_keyboard_callback(std::bind(&Pipeline::keyboard_callback, this, std::placeholders::_1,
                                                 std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
    }

    /// [GENERIC_TRACKING_SET_CAMERA_CALLBACK_BEGIN]
    // Connects filters.
    camera_->cd().add_callback([this](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        // Updates the frame generator if enabled.
        if (frame_generation_)
            frame_generation_->process_events(begin, end);

        // Processes pending commands.
        if (tracker_cmd_) {
            tracker_cmd_();
            tracker_cmd_ = nullptr;
        }

        const Metavision::EventCD *it_begin = begin;
        const Metavision::EventCD *it_end   = end;
        event_buffer_.clear();

        if (stc_algo_) {
            event_buffer_.reserve(std::distance(it_begin, it_end));

            const auto last = stc_algo_->process_events(it_begin, it_end, event_buffer_.begin());
            const auto size = std::distance(event_buffer_.begin(), last);

            it_begin = event_buffer_.data();
            it_end   = it_begin + size;
        }

        // Processes events.
        tracker_->process_events(it_begin, it_end);
    });
    /// [GENERIC_TRACKING_SET_CAMERA_CALLBACK_END]

    /// [GENERIC_TRACKING_SET_OUTPUT_CALLBACK_BEGIN]
    tracker_->set_output_callback(
        std::bind(&Pipeline::tracker_callback, this, std::placeholders::_1, std::placeholders::_2));
    /// [GENERIC_TRACKING_SET_OUTPUT_CALLBACK_END]

    // Ends the pipeline when the camera is stopped.
    camera_->add_status_change_callback([this](const Metavision::CameraStatus &status) {
        if (status == Metavision::CameraStatus::STOPPED) {
            std::lock_guard<std::mutex> lock(process_mutex_);
            is_processing_ = false;

            if (window_)
                window_->set_close_flag();

            process_cond_.notify_all();
        }
    });
}

/// Main function
int main(int argc, char *argv[]) {
    Pipeline pipeline;

    // Parses command line.
    if (!pipeline.parse_command_line(argc, argv))
        return 1;

    // Initializes the camera.
    if (!pipeline.initialize_camera())
        return 2;

    // Initializes the tracker.
    pipeline.initialize_tracker();

    // Starts the camera.
    if (!pipeline.start())
        return 3;

    // Waits until the end of the pipeline.
    pipeline.run();

    pipeline.stop();

    return 0;
}
