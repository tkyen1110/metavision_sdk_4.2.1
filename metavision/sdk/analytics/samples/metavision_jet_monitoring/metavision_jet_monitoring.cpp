/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <opencv2/core/types.hpp>
#include <string>

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/core/algorithms/on_demand_frame_generation_algorithm.h>
#include <metavision/sdk/analytics/algorithms/jet_monitoring_algorithm.h>
#include <metavision/sdk/analytics/configs/jet_monitoring_configs.h>
#include <metavision/sdk/analytics/utils/jet_monitoring_drawing_helper.h>
#include <metavision/sdk/analytics/utils/jet_monitoring_logger.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/ui/utils/mt_window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

#include "simple_video_writer.h"

namespace bpo = boost::program_options;
namespace bfs = boost::filesystem;

using timestamp         = Metavision::timestamp;
using NozzleOrientation = Metavision::JetMonitoringAlgorithmConfig::Orientation;

namespace Metavision {
// These two operator overloads are required for the NozzleOrientation enum to work with boost::program_options.
std::istream &operator>>(std::istream &in, NozzleOrientation &orientation) {
    std::string token;
    in >> token;
    if (token == "Down")
        orientation = NozzleOrientation::Down;
    else if (token == "Up")
        orientation = NozzleOrientation::Up;
    else if (token == "Left")
        orientation = NozzleOrientation::Left;
    else if (token == "Right")
        orientation = NozzleOrientation::Right;
    else
        in.setstate(std::ios_base::failbit);
    return in;
}

std::ostream &operator<<(std::ostream &out, const NozzleOrientation &orientation) {
    if (orientation == NozzleOrientation::Down)
        out << "Down";
    else if (orientation == NozzleOrientation::Up)
        out << "Up";
    else if (orientation == NozzleOrientation::Left)
        out << "Left";
    else if (orientation == NozzleOrientation::Right)
        out << "Right";
    else
        throw std::invalid_argument("Invalid nozzle orientation. Must be either Down, Up, Left or Right.");

    return out;
}
} // namespace Metavision

namespace std {
// This operator overload is required to set a default_value to ROIs.
std::ostream &operator<<(std::ostream &os, const std::vector<uint16_t> &vec) {
    for (auto item : vec)
        os << item << " ";
    return os;
}
} // namespace std

class Pipeline {
public:
    Pipeline() = default;

    ~Pipeline() = default;

    /// @brief Utility function to parse command line attributes
    bool parse_command_line(int argc, char *argv[]);

    /// @brief Initialize the camera
    bool initialize_camera();

    /// @brief Initialize the algorithm
    bool initialize_algorithm();

    /// @brief Start the camera
    bool start();

    /// @brief Stop the pipeline
    void stop();

    /// @brief Wait until the end of the file or until the exit of the display
    void run();

private:
    /// @brief Processing of the events coming from the camera
    void camera_callback(const Metavision::EventCD *begin, const Metavision::EventCD *end);

    /// @brief Callback called by the @ref Metavision::MTWindow when a key is pressed
    void keyboard_callback(Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods);

    void async_callback(const timestamp processing_ts, const size_t n_processed_events);

    void print_keys_help_message();

    void record_rawfile();

    // Camera parameters
    int width_, height_;
    std::string biases_file_ = "";
    std::string input_file_  = "";

    timestamp process_from_ = 0;  ///< Start time to process events and write the output video (in us)
    timestamp process_to_   = -1; ///< End time to process events and write the output video (in us)

    // Configs
    Metavision::JetMonitoringAlgorithmConfig algo_config_;
    Metavision::JetMonitoringAlarmConfig alarm_config_;
    Metavision::JetMonitoringLoggerConfig logger_config_;

    // Display parameters
    bool no_display_          = false; ///< If display data on the screen or not
    bool as_fast_as_possible_ = false; ///< If display as fast as possible
    std::string output_video_ = "";    ///< Filename to save the resulted video
    cv::Mat back_img_;                 ///< Current image

    // Last results
    timestamp ts_most_recent_;
    int last_count_ = 0;

    // Camera and detection ROIs
    std::vector<uint16_t> camera_roi_vec_;
    std::vector<uint16_t> detection_roi_vec_;
    cv::Rect camera_roi_;

    // Conditional variables to notify the end of the processing
    std::condition_variable process_cond_;
    std::mutex process_mutex_;
    volatile bool is_processing_ = true;

    // Algorithms
    std::unique_ptr<Metavision::Camera> camera_;                                           ///< Camera
    std::unique_ptr<Metavision::JetMonitoringAlgorithm> jet_monitoring_algo_;              ///< Jet Monitoring
    std::unique_ptr<Metavision::OnDemandFrameGenerationAlgorithm> events_frame_generator_; ///< Events Frame generator
    std::unique_ptr<Metavision::JetMonitoringDrawingHelper> jet_drawing_helper_;

    std::unique_ptr<SimpleVideoWriter> video_writer_; ///< Video writer
    std::unique_ptr<Metavision::MTWindow> window_;    ///< Display window
    std::unique_ptr<Metavision::JetMonitoringLogger> logger_;
};

bool Pipeline::parse_command_line(int argc, char *argv[]) {
    const std::string short_program_desc(
        "Code sample for Jet Monitoring on a stream of events from an event-based device or recorded data.\n");

    const std::string long_program_desc(
        short_program_desc +
        "This sample detects, counts, and timestamps the jets that are being dispensed.\n"
        "\n"
        "Please note that the GUI is displayed only when reading RAW files "
        "as it would not make sense to generate frames at such high frequency\n\n"
        "On the top left, you will see three lines:\n"
        "   - Time elapsed since the beginning of the app: this is the camera time in microseconds\n"
        "   - Current event rate in kEV/s. It varies depending on the activity\n"
        "   - Current jets count\n"
        "An arrow and several rectangles are displayed on the 'GUI':\n"
        "   - The arrow represents the direction in which the nozzle fires jets \n"
        "   - The largest red rectangle represents the --camera-roi, i.e. the area seen by the camera\n"
        "   - The small red rectangle represents the --detection-roi, i.e. the area where the algorithm looks for "
        "peaks in the event-rate\n"
        "   - The two blue rectangles that surround the --detection-roi are the areas used to monitor the background "
        "noise\n"
        "\n"
        "If set, alarms will warn you when there's something wrong.\n"
        "When the logging is enabled, please mind to choose a folder name that reflects the picopulse input settings.\n"
        "\n");

    bpo::options_description options_desc;
    bpo::options_description input_options("Input options");
    // clang-format off
    input_options.add_options()
        ("help,h", "Produce help message.")
        ("input-file,i",     bpo::value<std::string>(&input_file_)->default_value(""), "Path to input file. If not specified, the camera live stream is used.")
        ("biases,b",         bpo::value<std::string>(&biases_file_)->default_value(""), "Path to a biases file. If not specified, the camera will be configured with the default biases.")
        ("process-from,s",   bpo::value<timestamp>(&process_from_)->default_value(0), "Start time to process events (in us).")
        ("process-to,e",     bpo::value<timestamp>(&process_to_)->default_value(-1), "End time to process events (in us).")
        ;
    // clang-format on

    bpo::options_description roi_options("ROI options");
    // clang-format off
    roi_options.add_options()
        ("camera-roi",    bpo::value<std::vector<uint16_t>>(&camera_roi_vec_)->multitoken()->default_value(std::vector<uint16_t>{160, 160, 124, 93}), "Camera ROI [Left x, Top y, width, height]. Note that the nozzle orientation "
                                                                                                                                                      "doesn't modify or rotate this ROI, it just indicates the direction "
                                                                                                                                                      "in which the jets pass through this area.")
        ("detection-roi", bpo::value<std::vector<uint16_t>>(&detection_roi_vec_)->multitoken()->default_value(std::vector<uint16_t>{177, 197, 47, 20}),
                                                                                         "Detection ROI [Left x, Top y, width, height] must be large enough so that a 'nominal' jet is contained "
                                                                                         "in it. But not too large so that jet unrelated activity doesn't trigger count. Note that the nozzle orientation "
                                                                                         "doesn't modify or rotate this ROI, it just indicates the direction in which the jets pass through this area.")
        ;
    // clang-format on

    bpo::options_description algo_options("Jet Monitoring Algorithm options");
    // clang-format off
    algo_options.add_options()
        ("nozzle-orientation",        bpo::value<NozzleOrientation>(&algo_config_.nozzle_orientation)->default_value(NozzleOrientation::Right),
                                      "Nozzle orientation in the image reference frame. Jets are moving either upwards, downwards, leftwards or rightwards: Up, Down, Left, Right.")
        ("time-step-us",              bpo::value<int>(&algo_config_.time_step_us)->default_value(50),          "Time step, update period of the monitoring (in us).")
        ("accumulation-time-us",      bpo::value<int>(&algo_config_.accumulation_time_us)->default_value(500), "Period over which to accumulate events. This must be set depending on the cycle. It must be slightly lower than the input cycle.")
        ("counting-th-up",            bpo::value<int>(&algo_config_.th_up_kevps)->default_value(50),           "Minimum activity to trigger a jet count, in kev/s. If too high, jets may be missed (activity is never reached).")
        ("counting-th-up-delay-us",   bpo::value<timestamp>(&algo_config_.th_up_delay_us)->default_value(100), "Filter time to confirm the beginning of a jet (in us).")
        ("counting-th-down",          bpo::value<int>(&algo_config_.th_down_kevps)->default_value(10),         "Lower bound activity that defines the end of a jet, in kev/s. If too low, jets may be missed (the jet never ends from the algorithm point of view).")
        ("counting-th-down-delay-us", bpo::value<timestamp>(&algo_config_.th_down_delay_us)->default_value(0), "Filter time to confirm the end of a jet (in us).")
        ;
    // clang-format on

    bpo::options_description alarm_options("Alarm options");
    // clang-format off
    alarm_options.add_options()
        ("alarm-on-count",       bpo::bool_switch(&alarm_config_.alarm_on_count),"If true, an alarm will be raised if jets are detected above the --max-expected-count.")
        ("max-expected-count",   bpo::value<int>(&alarm_config_.max_expected_count)->default_value(0), "Maximum expected number of jets.")
        ("alarm-on-cycle",       bpo::bool_switch(&alarm_config_.alarm_on_cycle), "If true, an alarm will be raised if cycle time (time between jets) is outside the specified tolerance.")
        ("expected-cycle-ms",    bpo::value<float>(&alarm_config_.expected_cycle_ms)->default_value(0), "Expected cycle time (in ms). If set to 0, no alarms will be generated for jet timing.")
        ("cycle-tol-percentage", bpo::value<float>(&alarm_config_.cycle_tol_percentage)->default_value(10), "Cycle tolerance, in percentage. If the time between two successive jets is off the --expected-cycle-ms "
                                                                                                                     "by more than this percentage, an alarm will be raised.")
        ;
    // clang-format on

    bpo::options_description logger_options("Logger options");
    // clang-format off
    logger_options.add_options()
        ("enable-logging",        bpo::bool_switch(&logger_config_.enable_logging), "Enable logging. ")
        ("log-output-dir",        bpo::value<std::string>(&logger_config_.log_out_dir)->default_value("/tmp/jet_monitoring"), "Log output dir. Each trigger will create a subdirectory on it.")
        ("log-history-length-ms", bpo::value<int>(&logger_config_.log_history_length_ms)->default_value(20), "Duration of each log dump, in ms.")
        ("log-dump-delay-ms",     bpo::value<int>(&logger_config_.log_dump_delay_ms)->default_value(0), "Wait for this duration before dumping the log, to keep some log data after the trigger."
                                                                                                        "Keep it smaller than the history length to make sure the trigger is included in the log.")
        ("dump-log-at-exit",      bpo::bool_switch(&logger_config_.dump_at_exit), "Dump log when exiting the application.")
        ("log-jet-video",         bpo::bool_switch(&logger_config_.log_jet_video), "Combine the images of each jet to create a video of an average jet.")
        ("log-jets-event-rate",   bpo::bool_switch(&logger_config_.log_jets_evt_rate), "Dump a file containing the event rate of each jet.")
        ;
    // clang-format on

    bpo::options_description gui_options("GUI options");
    // clang-format off
    gui_options.add_options()
        ("no-display",  bpo::bool_switch(&no_display_), "Disable the GUI when reading a file (no effect with a live camera where GUI is already disabled).")
        ("out-video,o", bpo::value<std::string>(&output_video_), "Path to an output AVI file to save the resulting slow motion video. A frame is generated after each process of the algorithm. The video "
                                                                 "will be written only for processed events. When the display is disabled, i.e. either with a live camera or when --no-display "
                                                                 "has been specified, frames are not generated, so the video can't be generated either.")
        ;
    // clang-format on

    options_desc.add(input_options)
        .add(roi_options)
        .add(algo_options)
        .add(alarm_options)
        .add(logger_options)
        .add(gui_options);

    bpo::variables_map vm;
    try {
        bpo::store(bpo::command_line_parser(argc, argv).options(options_desc).run(), vm);
        bpo::notify(vm);
    } catch (bpo::error &e) {
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
    const bool is_offline = !input_file_.empty();
    if (is_offline && no_display_) {
        as_fast_as_possible_ = true;
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

    if (logger_config_.log_history_length_ms <= 0) {
        MV_LOG_ERROR() << "The log history length must be strictly positive:" << logger_config_.log_history_length_ms;
        return false;
    }

    if (logger_config_.log_dump_delay_ms < 0) {
        MV_LOG_ERROR() << "The log dump delay must be positive:" << logger_config_.log_history_length_ms;
        return false;
    }

    if (logger_config_.log_dump_delay_ms >= logger_config_.log_history_length_ms) {
        MV_LOG_ERROR() << "The log dump delay can't be larger than the log history length."
                       << "Otherwise the trigger isn't included in the log dump:" << logger_config_.log_dump_delay_ms
                       << ">=" << logger_config_.log_history_length_ms;
        return false;
    }

    if (camera_roi_vec_.size() != 4) {
        MV_LOG_ERROR() << Metavision::Log::no_space
                       << "The camera ROI [x, y, width, height] must contain 4 elements, not " << camera_roi_vec_.size()
                       << ".";
        return false;
    }

    if (detection_roi_vec_.size() != 4) {
        MV_LOG_ERROR() << Metavision::Log::no_space
                       << "The detection ROI [x, y, width, height] must contain 4 elements, not "
                       << detection_roi_vec_.size() << ".";
        return false;
    }

    if (camera_roi_vec_[0] > detection_roi_vec_[0]                                                   // Left
        || camera_roi_vec_[1] > detection_roi_vec_[1]                                                // Top
        || (detection_roi_vec_[0] + detection_roi_vec_[2] > camera_roi_vec_[0] + camera_roi_vec_[2]) // Right
        || (detection_roi_vec_[1] + detection_roi_vec_[3] > camera_roi_vec_[1] + camera_roi_vec_[3]) // Bottom
    ) {
        MV_LOG_ERROR() << "The camera ROI must contain the detection ROI.";
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

bool Pipeline::initialize_camera() {
    camera_roi_ = cv::Rect(camera_roi_vec_[0], camera_roi_vec_[1], camera_roi_vec_[2], camera_roi_vec_[3]);
    algo_config_.detection_roi =
        cv::Rect(detection_roi_vec_[0], detection_roi_vec_[1], detection_roi_vec_[2], detection_roi_vec_[3]);
    MV_LOG_INFO() << "Camera ROI    :" << camera_roi_;
    MV_LOG_INFO() << "Detection ROI :" << algo_config_.detection_roi;

    // If the filename is set, then read from the file
    if (input_file_ != "") {
        try {
            camera_ = std::make_unique<Metavision::Camera>(Metavision::Camera::from_file(
                input_file_, Metavision::FileConfigHints().real_time_playback(!as_fast_as_possible_)));
        } catch (Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return false;
        }
        // Otherwise, set the input source to the fist available camera
    } else {
        try {
            camera_ = std::make_unique<Metavision::Camera>(Metavision::Camera::from_first_available());

            if (biases_file_ != "") {
                camera_->biases().set_from_file(biases_file_);
            }
            Metavision::Roi::Window roi_rect;
            roi_rect.x      = camera_roi_.x;
            roi_rect.y      = camera_roi_.y;
            roi_rect.width  = camera_roi_.width;
            roi_rect.height = camera_roi_.height;

            std::stringstream ss;
            ss << "Setting camera ROI: " << roi_rect.x << ", " << roi_rect.y << ", " << roi_rect.width << ", "
               << roi_rect.height;
            MV_LOG_INFO() << ss.str();
            camera_->roi().set(roi_rect);

        } catch (Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return false;
        }
    }

    // Add camera runtime error callback
    camera_->add_runtime_error_callback([](const Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); });

    return true;
}

void Pipeline::camera_callback(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
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

    jet_monitoring_algo_->process_events(begin, end);
    if (logger_)
        logger_->process_events(begin, end);

    if (events_frame_generator_)
        events_frame_generator_->process_events(begin, end);
}

void Pipeline::record_rawfile() {
    if (!logger_config_.enable_logging)
        return;

    auto t  = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    camera_->stop_recording();
    camera_->start_recording((bfs::path(logger_config_.log_out_dir) / ("log_" + oss.str())).string() + ".raw");
}

void Pipeline::print_keys_help_message() {
    MV_LOG_INFO() << "---------------------------------------------------";
    MV_LOG_INFO() << "Jet monitoring";
    MV_LOG_INFO();
    MV_LOG_INFO() << "Press       -> To:";
    MV_LOG_INFO() << " * ESC or q -> Quit the application.";
    MV_LOG_INFO() << " * 0 or o   -> Reset counters/timers.";
    MV_LOG_INFO() << " * l        -> Dump Logs to disk.";
    MV_LOG_INFO() << " * h        -> Print this message.";
    MV_LOG_INFO() << "---------------------------------------------------";
}

void Pipeline::keyboard_callback(Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
    if (action == Metavision::UIAction::RELEASE) {
        switch (key) {
        case Metavision::UIKeyEvent::KEY_Q:
        case Metavision::UIKeyEvent::KEY_ESCAPE: {
            MV_LOG_INFO() << "Exiting application...";
            std::lock_guard<std::mutex> lock(process_mutex_);
            is_processing_ = false;
        }
            process_cond_.notify_all();
            window_->set_close_flag();
            break;

        case Metavision::UIKeyEvent::KEY_0:
        case Metavision::UIKeyEvent::KEY_O:
            MV_LOG_INFO() << " ============= Reset count ==============";
            jet_monitoring_algo_->reset_state();
            last_count_ = 0;
            break;

        case Metavision::UIKeyEvent::KEY_L:
            if (logger_) {
                MV_LOG_INFO() << " ============ Manual log dump ===========";
                logger_->schedule_dump(ts_most_recent_, "manual");
                record_rawfile();
            }
            break;

        case Metavision::UIKeyEvent::KEY_H:
            print_keys_help_message();
            break;

        default:
            break;
        }
    }
}

bool Pipeline::initialize_algorithm() {
    const auto &geometry = camera_->geometry();
    width_               = geometry.width();
    height_              = geometry.height();
    cv::Size sensor_size(width_, height_);

    // Jet Monitoring Algorithm and its callbacks
    jet_monitoring_algo_ = std::make_unique<Metavision::JetMonitoringAlgorithm>(algo_config_, alarm_config_);
    jet_monitoring_algo_->set_on_jet_callback([&](const auto &jet_in) {
        last_count_ = jet_in.count;
        MV_LOG_INFO() << "[o] ts: " << jet_in.t << ", jet_count: " << jet_in.count << ", delta_t: "
                      << ((jet_in.previous_jet_dt < 0) ? "N/A" : std::to_string(jet_in.previous_jet_dt) + "us.");
    });

    jet_monitoring_algo_->set_on_alarm_callback([&](const auto &alarm) {
        MV_LOG_INFO() << "[a] ts: " << alarm.t << ", (trigger ts: " << alarm.alarm_ts << ") " << alarm.info;
        if (logger_)
            logger_->process_alarm(alarm);
    });

    jet_monitoring_algo_->set_on_async_callback([&](const timestamp processing_ts, const size_t n_processed_events) {
        async_callback(processing_ts, n_processed_events);
    });

    // Logger
    if (logger_config_.enable_logging) {
        logger_ = std::make_unique<Metavision::JetMonitoringLogger>(sensor_size, camera_roi_, algo_config_,
                                                                    alarm_config_, logger_config_);
        jet_monitoring_algo_->set_on_slice_callback([&](const auto &slice_data) { logger_->log(slice_data); });
    }

    // Display
    if (!output_video_.empty() || !no_display_) {
        jet_drawing_helper_ = std::make_unique<Metavision::JetMonitoringDrawingHelper>(
            camera_roi_, algo_config_.detection_roi, algo_config_.nozzle_orientation);
        events_frame_generator_ = std::make_unique<Metavision::OnDemandFrameGenerationAlgorithm>(
            sensor_size.width, sensor_size.height, algo_config_.accumulation_time_us);

        if (!output_video_.empty()) {
            video_writer_ = std::make_unique<SimpleVideoWriter>(sensor_size.width, sensor_size.height,
                                                                algo_config_.time_step_us, 30.f, output_video_);
            video_writer_->set_write_range(process_from_, process_to_);
        }

        if (!no_display_) {
            window_ = std::make_unique<Metavision::MTWindow>("Jet Monitoring Sample. Press 'q' or ESCAPE to exit.",
                                                             width_, height_, Metavision::BaseWindow::RenderMode::BGR);
            window_->set_keyboard_callback(std::bind(&Pipeline::keyboard_callback, this, std::placeholders::_1,
                                                     std::placeholders::_2, std::placeholders::_3,
                                                     std::placeholders::_4));

            print_keys_help_message();
        }
    }

    // Camera Callback
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

    if (logger_)
        record_rawfile();

    return true;
}

void Pipeline::async_callback(const timestamp processing_ts, const size_t n_processed_events) {
    if (is_processing_) {
        ts_most_recent_ = processing_ts;

        if (events_frame_generator_) {
            events_frame_generator_->generate(processing_ts, back_img_);
            const int er_kevps = static_cast<int>((1000 * n_processed_events) / algo_config_.accumulation_time_us);
            jet_drawing_helper_->draw(processing_ts, last_count_, er_kevps, back_img_);
        }

        if (video_writer_)
            video_writer_->write_frame(processing_ts, back_img_);

        if (window_)
            window_->show_async(back_img_);
    }
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

    // Initialize trackers
    if (!pipeline.initialize_algorithm())
        return 3;

    // Keep the start time of execution
    const auto start = std::chrono::high_resolution_clock::now();

    // Start the camera
    if (!pipeline.start())
        return 4;

    // Wait until the end of the pipeline
    pipeline.run();

    pipeline.stop();

    // Estimate the total time of execution
    const auto end     = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    MV_LOG_INFO() << "Ran in" << static_cast<float>(elapsed.count()) / 1000.f << "s";
    return 0;
}
