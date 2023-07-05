/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// Example of using METAVISION SDK for counting objects

#include <mutex>
#include <condition_variable>
#include <boost/program_options.hpp>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/analytics/algorithms/counting_algorithm.h>
#include <metavision/sdk/analytics/utils/mono_counting_status.h>
#include <metavision/sdk/analytics/utils/counting_calibration.h>
#include <metavision/sdk/analytics/utils/counting_drawing_helper.h>
#include <metavision/sdk/base/events/event2d.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/cv/algorithms/activity_noise_filter_algorithm.h>
#include <metavision/sdk/cv/algorithms/transpose_events_algorithm.h>
#include <metavision/sdk/core/algorithms/polarity_filter_algorithm.h>
#include <metavision/sdk/core/algorithms/on_demand_frame_generation_algorithm.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

#include "simple_video_writer.h"

namespace boost_po = boost::program_options;

using timestamp = Metavision::timestamp;

class Pipeline {
public:
    Pipeline() = default;

    ~Pipeline() = default;

    /// @brief Utility function to parse command line attributes
    bool parse_command_line(int argc, char *argv[]);

    /// @brief Initialize the camera
    bool initialize_camera();

    /// @brief Initialize the counters
    bool initialize_counters();

    /// @brief Start the camera
    bool start();

    /// @brief Stop the pipeline
    void stop();

    /// @brief Wait until the end of the file or until the exit of the display
    void run();

private:
    /// @brief Handle current time
    void current_time_callback(timestamp ts);

    /// @brief Function that is called every N number of counted objects. N is a parameter that could be chosen (see
    /// notification sampling parameter)
    void increment_callback(const timestamp ts, const int count);

    /// @brief Function that is called every time no object is counted during a certain time. This time could be chosen
    /// with the parameter inactivity time.
    void inactivity_callback(const timestamp ts, const timestamp last_count_ts, const int count);

    /// @brief Processing of the counting output
    void counting_callback(const std::pair<timestamp, Metavision::MonoCountingStatus> &counting_result);

    /// @brief Processing of the events coming from the camera
    void camera_callback(const Metavision::EventCD *begin, const Metavision::EventCD *end);

    /// @brief Callback called by the @ref Metavision::Window when a key is pressed
    void keyboard_callback(Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods);

    /// CountingAlgorithm
    int min_y_line_;               ///< Min y to place the first line
    int max_y_line_;               ///< Max y to place the last line
    int num_lines_;                ///< Nbr of lines for counting between min-y and max-y
    float object_min_size_;        ///< Approximate minimum size of an object to count (in mm)
    float object_average_speed_;   ///< Approximate average speed of an object to count (in m/s)
    float distance_object_camera_; ///< Average distance between the flow of objects to count and the camera (in mm)

    /// Notifications
    int last_count_callback_ = 0;
    int inactivity_time_;       ///< Time of inactivity in us (no counter increment) to be notified
    int notification_sampling_; ///< Minimal number of counted objects between each notification
    timestamp last_ts_          = -1;
    timestamp last_ts_callback_ = -1;

    /// Filtering
    std::vector<Metavision::EventCD> buffer_filters_;
    short polarity_;              ///< Process only events of this polarity
    timestamp activity_time_ths_; ///< Length of the time window for activity filtering (in us)

    /// Camera parameters
    std::string filename_    = "";
    std::string biases_file_ = "";

    /// Display parameters
    bool no_display_                 = true;  ///< If we display data on the screen or not
    bool replay_as_fast_as_possible_ = false; ///< If we replay the recording as fast as possible
    std::string output_video_        = "";    ///< Filename to save the resulted video
    cv::Mat back_img_;                        ///< Current image

    /// Visualization
    bool transpose_axis_; ///< Set to true to rotate the camera 90 degrees clockwise in case of particles moving
                          ///< horizontally in FOV

    // Conditional variables to notify the end of the processing
    std::condition_variable process_cond_;
    std::mutex process_mutex_;
    volatile bool is_processing_ = true;

    /// Time
    timestamp process_from_ = 0;  ///< Start time to process events and write the output video (in us)
    timestamp process_to_   = -1; ///< End time to process events and write the output video (in us)

    std::unique_ptr<Metavision::Camera> camera_;                                            ///< Camera
    std::unique_ptr<Metavision::CountingAlgorithm> counting_algo_;                          ///< Counting algorithm
    std::unique_ptr<Metavision::CountingDrawingHelper> counting_drawing_helper_;            ///< Counting drawing helper
    std::unique_ptr<Metavision::OnDemandFrameGenerationAlgorithm> events_frame_generation_; ///< Events Frame generator
    std::unique_ptr<SimpleVideoWriter> video_writer_;                                       ///< Video writer
    std::unique_ptr<Metavision::Window> window_;                                            ///< Display window
    std::unique_ptr<Metavision::PolarityFilterAlgorithm> polarity_filter_;                  ///< Filter by polarity
    std::unique_ptr<Metavision::TransposeEventsAlgorithm> transpose_events_filter_;         ///< Transpose X/Y on events
    std::unique_ptr<Metavision::ActivityNoiseFilterAlgorithm<>> activity_noise_filter_;     ///< Filter noisy events
};

bool Pipeline::initialize_camera() {
    // If the filename is set, then read from the file
    if (!filename_.empty()) {
        try {
            camera_ = std::make_unique<Metavision::Camera>(Metavision::Camera::from_file(
                filename_, Metavision::FileConfigHints().real_time_playback(!replay_as_fast_as_possible_)));
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
    const std::string short_program_desc(
        "Code sample for counting algorithm on a stream of events from an event-based device or recorded data.\n");

    const std::string long_program_desc(
        short_program_desc +
        "By default, this samples uses only OFF events and assumes that the objects are moving vertically in FOV.\n"
        "In case of different configuration, the default parameters should be adjusted.\n"
        "Please note that the GUI is displayed only when reading files "
        "as it would not make sense to generate frames at such high frequency\n\n"
        "Press 'q' or Escape key to leave the program.\n"
        "Press 'r' to reset the counter.\n"
        "Press 'p' to increase (+1) notification sampling (number of objects to be counted between each "
        "notification).\n"
        "Press 'm' to decrease (-1) notification sampling (number of objects to be counted between each "
        "notification).\n");

    boost_po::options_description options_desc;
    boost_po::options_description base_options("Base options");
    // clang-format off
    base_options.add_options()
        ("help,h", "Produce help message.")
        ("input-file,i",       boost_po::value<std::string>(&filename_), "Path to input file. If not specified, the camera live stream is used.")
        ("biases,b",           boost_po::value<std::string>(&biases_file_), "Path to a biases file. If not specified, the camera will be configured with the default biases.")
        ("process-from,s",     boost_po::value<timestamp>(&process_from_), "Start time to process events (in us).")
        ("process-to,e",       boost_po::value<timestamp>(&process_to_), "End time to process events (in us).")
        ;
    // clang-format on

    boost_po::options_description filter_options("Filtering options");
    // clang-format off
    filter_options.add_options()
        ("activity-ths", boost_po::value<timestamp>(&activity_time_ths_)->default_value(0), "Length of the time window for activity filtering (in us).")
        ("polarity,p",   boost_po::value<short>(&polarity_)->default_value(0), "Which event polarity to process : 0 (OFF), 1 (ON), -1 (ON & OFF). By default it uses only OFF events.")
        ("rotate,r",     boost_po::bool_switch(&transpose_axis_)->default_value(false), "Rotate the camera 90 degrees clockwise in case of particles moving horizontally in FOV.")
        ;
    // clang-format on

    boost_po::options_description calib_options("Calibration options");
    // clang-format off
    calib_options.add_options()
        ("object-min-size",        boost_po::value<float>(&object_min_size_)->default_value(6.), "Approximate largest dimension of the smallest object (in millimeters).")
        ("object-average-speed",   boost_po::value<float>(&object_average_speed_)->default_value(5.), "Approximate average speed of an object to count in meters per second.")
        ("distance-object-camera", boost_po::value<float>(&distance_object_camera_)->default_value(300.), "Average distance between the flow of objects to count and the camera (distance in millimeters).")
        ;
    // clang-format on

    boost_po::options_description algo_options("Algorithm options");
    // clang-format off
    algo_options.add_options()
        ("num-lines,n", boost_po::value<int>(&num_lines_)->default_value(4), "Number of lines for counting between min-y and max-y.")
        ("min-y",       boost_po::value<int>(&min_y_line_)->default_value(150), "Ordinate at which to place the first line counter.")
        ("max-y",       boost_po::value<int>(&max_y_line_)->default_value(330), "Ordinate at which to place the last line counter.")
        ;
    // clang-format on

    boost_po::options_description outcome_options("Outcome options");
    // clang-format off
    outcome_options.add_options()
        ("no-display",            boost_po::bool_switch(&no_display_), "Disable the GUI when reading a file (no effect with a live camera where GUI is already disabled).")
        ("out-video,o",           boost_po::value<std::string>(&output_video_), "Path to an output AVI file to save the resulting slow motion video. A frame is generated after each process of the algorithm. The video "
                                                                                "will be written only for processed events. When the display is disabled, i.e. either with a live camera or when --no-display "
                                                                                "has been specified, frames are not generated, so the video can't be generated either.")
        ("notification-sampling", boost_po::value<int>(&notification_sampling_)->default_value(1), "Minimal number of counted objects between each notification.")
        ("inactivity-time",       boost_po::value<int>(&inactivity_time_)->default_value(static_cast<int>(1e6)), "Time of inactivity in us (no counter increment) to be notified.")
        ;
    // clang-format on

    options_desc.add(base_options).add(calib_options).add(algo_options).add(filter_options).add(outcome_options);

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

    // When running offline on a recording, we want to replay data as fast as possible when the display is disabled
    const bool is_offline = !filename_.empty();
    if (is_offline && no_display_) {
        replay_as_fast_as_possible_ = true;
    }

    // When running from a live camera, disable the display
    if (!is_offline) {
        MV_LOG_INFO()
            << "Display is automatically disabled with a live camera, since it doesn't make sense to generate "
               "frames at such a high frequency.";
        no_display_ = true;
    }

    if (!output_video_.empty() && no_display_) {
        MV_LOG_ERROR() << "Try to generate an output video whereas the display is disabled";
        return false;
    }

    // Check polarity
    if (polarity_ != 1 && polarity_ != 0 && polarity_ != -1) {
        MV_LOG_ERROR() << "The polarity is not valid:" << polarity_;
        return false;
    }

    // Check positions of line counters
    if (max_y_line_ <= min_y_line_) {
        MV_LOG_ERROR() << "The range of y-positions for the line counters is not valid: " << Metavision::Log::no_space
                       << "[" << min_y_line_ << ", " << max_y_line_ << "].";
        return false;
    }

    MV_LOG_INFO() << long_program_desc;

    return true;
}

bool Pipeline::start() {
    const bool started = camera_->start();
    if (!started) {
        MV_LOG_ERROR() << "The camera could not be started.";
    }
    return started;
}

// Stop the camera when finished
void Pipeline::stop() {
    try {
        camera_->stop();
    } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
}

void Pipeline::run() {
    if (window_) {
        while (!window_->should_close()) {
            // we sleep the main thread a bit to avoid using 100% of a CPU's core
            static constexpr std::int64_t kSleepPeriodMs = 20;

            Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
        }
    } else {
        // Wait until the end of the file
        std::unique_lock<std::mutex> lock(process_mutex_);
        process_cond_.wait(lock, [this] { return !is_processing_; });
    }
}

void Pipeline::current_time_callback(timestamp ts) {
    if (last_ts_ < 0) {
        last_ts_ = ts;
    } else if (ts > last_ts_) {
        MV_LOG_INFO() << "Current time:" << ts;
        last_ts_ += static_cast<timestamp>(1e6);
    }
}

void Pipeline::increment_callback(const timestamp ts, const int count) {
    if (count >= last_count_callback_ + notification_sampling_) {
        last_count_callback_ += ((count - last_count_callback_) / notification_sampling_) * notification_sampling_;
        MV_LOG_INFO() << "At" << ts << "counter is" << count;
    }
}

void Pipeline::inactivity_callback(const timestamp ts, const timestamp last_count_ts, const int count) {
    last_ts_callback_ = std::max(last_count_ts, last_ts_callback_);
    if (ts >= last_ts_callback_ + inactivity_time_) {
        last_ts_callback_ += ((ts - last_ts_callback_) / inactivity_time_) * inactivity_time_;
        MV_LOG_INFO() << "At" << ts << "inactivity period.";
    }
}

/// [COUNTING_CALLBACK_BEGIN]
void Pipeline::counting_callback(const std::pair<timestamp, Metavision::MonoCountingStatus> &counting_result) {
    if (!is_processing_)
        return;

    const timestamp &ts            = counting_result.first;
    const timestamp &last_count_ts = counting_result.second.last_count_ts;
    const int count                = counting_result.second.global_counter;

    if (counting_drawing_helper_) {
        events_frame_generation_->generate(ts, back_img_);
        counting_drawing_helper_->draw(ts, count, back_img_);

        if (video_writer_)
            video_writer_->write_frame(ts, back_img_);

        if (window_)
            window_->show(back_img_);
    }

    current_time_callback(ts);
    increment_callback(ts, count);
    inactivity_callback(ts, last_count_ts, count);
}
/// [COUNTING_CALLBACK_END]

/// Function that applies a filter on events if it's not a nullptr.
/// The boolean parameter is_first indicates whether the filter should be applied on the raw events
/// or on the events that have already been filtered through other filters.
template<class AlgoPtr, class InputIt, class FilteredIt>
inline void apply_filter_if_enabled(const AlgoPtr &algo, InputIt &begin, InputIt &end,
                                    std::vector<FilteredIt> &output_buffer, bool &is_first) {
    if (algo) {
        if (is_first) {
            assert(begin != nullptr && end != nullptr);
            output_buffer.clear();
            algo->process_events(begin, end, std::back_inserter(output_buffer));
            is_first = false; // the next filters will have to use output_buffer instead of begin and end
            begin    = nullptr;
            end      = nullptr;
        } else {
            auto end_it = algo->process_events(output_buffer.cbegin(), output_buffer.cend(), output_buffer.begin());
            output_buffer.resize(std::distance(output_buffer.begin(), end_it));
        }
    }
}

/// Function that passes the events to an algorithm.
/// The boolean parameter is_first indicates whether the raw events have been filtered or not.
/// If is_first is set to true, this means that there were no filters applied to the events.
template<class AlgoPtr, class InputIt, class FilteredIt>
inline void apply_algorithm_if_enabled(const AlgoPtr &algo, InputIt &begin, InputIt &end,
                                       std::vector<FilteredIt> &filtered_buffer, const bool &is_first) {
    if (algo) {
        if (is_first) {
            assert(begin != nullptr && end != nullptr);
            algo->process_events(begin, end);
        } else
            algo->process_events(filtered_buffer.cbegin(), filtered_buffer.cend());
    }
}

/// [CAMERA_CALLBACK_BEGIN]
void Pipeline::camera_callback(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
    if (!is_processing_)
        return;

    // Adjust iterators to make sure we only process a given range of timestamps [process_from_, process_to_]
    // Get iterator to the first element greater or equal than process_from_
    begin = std::lower_bound(begin, end, process_from_,
                             [](const Metavision::EventCD &ev, timestamp ts) { return ev.t < ts; });

    // Get iterator to the first element greater than process_to_
    if (process_to_ >= 0)
        end = std::lower_bound(begin, end, process_to_,
                               [](const Metavision::EventCD &ev, timestamp ts) { return ev.t <= ts; });
    if (begin == end)
        return;

    /// Apply filters
    bool is_first = true;
    apply_filter_if_enabled(polarity_filter_, begin, end, buffer_filters_, is_first);
    apply_filter_if_enabled(transpose_events_filter_, begin, end, buffer_filters_, is_first);
    apply_filter_if_enabled(activity_noise_filter_, begin, end, buffer_filters_, is_first);

    /// Process filtered events
    apply_algorithm_if_enabled(events_frame_generation_, begin, end, buffer_filters_, is_first);
    apply_algorithm_if_enabled(counting_algo_, begin, end, buffer_filters_, is_first);
}
/// [CAMERA_CALLBACK_END]

void Pipeline::keyboard_callback(Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
    if (action == Metavision::UIAction::RELEASE) {
        switch (key) {
        case Metavision::UIKeyEvent::KEY_Q:
        case Metavision::UIKeyEvent::KEY_ESCAPE: {
            std::lock_guard<std::mutex> lock(process_mutex_);
            is_processing_ = false;
            process_cond_.notify_all();
        }
            window_->set_close_flag();
            break;
        case Metavision::UIKeyEvent::KEY_R:
            MV_LOG_INFO() << "Reset counter";
            counting_algo_->reset_counters();
            break;
        case Metavision::UIKeyEvent::KEY_P:
            ++notification_sampling_;
            MV_LOG_INFO() << "Setting notification sampling to" << notification_sampling_;
            break;
        case Metavision::UIKeyEvent::KEY_M:
            if (notification_sampling_ >= 2) {
                --notification_sampling_;
                MV_LOG_INFO() << "Setting notification sampling to" << notification_sampling_;
            }
            break;
        default:
            break;
        }
    }
}

bool Pipeline::initialize_counters() {
    /// Camera Geometry
    auto &geometry = camera_->geometry();
    int sensor_width, sensor_height;
    if (transpose_axis_) {
        transpose_events_filter_ = std::make_unique<Metavision::TransposeEventsAlgorithm>();
        sensor_width             = geometry.height();
        sensor_height            = geometry.width();
    } else {
        sensor_width  = geometry.width();
        sensor_height = geometry.height();
    }

    /// Positions of the line counters
    if (min_y_line_ < 0) {
        MV_LOG_ERROR() << "min-y should be positive.";
        MV_LOG_ERROR() << "min-y passed: " << min_y_line_;
        return false;
    }
    if (max_y_line_ > sensor_height) {
        MV_LOG_ERROR() << "max-y should be smaller than the sensor's height (" << sensor_height << ").";
        MV_LOG_ERROR() << "max-y passed: " << max_y_line_;
        return false;
    }
    std::vector<int> rows; // vector to fill with line counters ordinates
    rows.reserve(num_lines_);
    const int y_line_step = (max_y_line_ - min_y_line_) / (num_lines_ - 1);
    for (int i = 0; i < num_lines_; ++i) {
        const int line_ordinate = min_y_line_ + y_line_step * i;
        rows.push_back(line_ordinate);
    }

    // Activity Noise Filter
    if (activity_time_ths_ != 0) {
        activity_noise_filter_ = std::make_unique<Metavision::ActivityNoiseFilterAlgorithm<>>(
            sensor_width, sensor_height, activity_time_ths_);
    }

    if (polarity_ >= 0) {
        polarity_filter_ = std::make_unique<Metavision::PolarityFilterAlgorithm>(polarity_);
    }

    /// [COUNTING_CALIBRATION_BEGIN]
    // CALIBRATION
    const auto calib_results = Metavision::CountingCalibration::calibrate(
        sensor_width, sensor_height, object_min_size_, object_average_speed_, distance_object_camera_);
    // CountingAlgorithm
    counting_algo_ = std::make_unique<Metavision::CountingAlgorithm>(
        sensor_width, sensor_height, calib_results.cluster_ths, calib_results.accumulation_time_us);
    /// [COUNTING_CALIBRATION_END]

    for (const int &row : rows) // Add lines
        counting_algo_->add_line_counter(row);

    if (!output_video_.empty() || !no_display_) {
        counting_drawing_helper_ = std::make_unique<Metavision::CountingDrawingHelper>(rows);
        events_frame_generation_ = std::make_unique<Metavision::OnDemandFrameGenerationAlgorithm>(
            sensor_width, sensor_height, calib_results.accumulation_time_us);
    }

    if (!output_video_.empty()) {
        video_writer_ = std::make_unique<SimpleVideoWriter>(sensor_width, sensor_height,
                                                            calib_results.accumulation_time_us, 30.f, output_video_);
        video_writer_->set_write_range(process_from_, process_to_);
    }

    if (!no_display_) {
        window_ =
            std::make_unique<Metavision::Window>("Counting Sample. Press 'q' or Escape key to exit.", sensor_width,
                                                 sensor_height, Metavision::BaseWindow::RenderMode::BGR);
        // Notifies the pipeline when the window is exited.
        window_->set_keyboard_callback(std::bind(&Pipeline::keyboard_callback, this, std::placeholders::_1,
                                                 std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
    }

    counting_algo_->set_output_callback(
        [this](const std::pair<timestamp, Metavision::MonoCountingStatus> &counting_result) {
            counting_callback(counting_result);
        });

    camera_->cd().add_callback(
        [this](const Metavision::EventCD *begin, const Metavision::EventCD *end) { camera_callback(begin, end); });

    // Stops the pipeline when the camera is stopped
    camera_->add_status_change_callback([this](const Metavision::CameraStatus &status) {
        if (status == Metavision::CameraStatus::STOPPED) {
            std::lock_guard<std::mutex> lock(process_mutex_);
            is_processing_ = false;

            if (window_)
                window_->set_close_flag();

            process_cond_.notify_all();
        }
    });

    return true;
}

/// Main function
int main(int argc, char *argv[]) {
    Pipeline pipeline;

    // Parse command line
    if (!pipeline.parse_command_line(argc, argv))
        return 1;

    // Initialize the camera
    if (!pipeline.initialize_camera())
        return 2;

    // Initialize counters
    if (!pipeline.initialize_counters())
        return 3;

    // Start the camera
    if (!pipeline.start())
        return 4;

    // Wait until the end of the pipeline
    pipeline.run();

    pipeline.stop();

    return 0;
}
