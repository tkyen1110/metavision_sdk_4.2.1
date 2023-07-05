/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <boost/program_options.hpp>
#include <memory>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/utils/sdk_log.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/cv/algorithms/transpose_events_algorithm.h>
#include <metavision/sdk/ui/utils/event_loop.h>

#include "jet_monitoring_calibration_gui.h"

namespace bpo = boost::program_options;

class Pipeline {
public:
    /// @brief Parses command line arguments
    bool parse_command_line(int argc, char *argv[]);

    /// @brief Initializes the Camera, the Window and the Event Frame Generator
    bool init();

    /// @brief Processes event-stream and reacts to mouse and key-events
    bool run();

private:
    Metavision::Camera camera_; ///< Camera

    std::unique_ptr<Metavision::JetMonitoringCalibrationGUI> gui_;                  ///< Graphical User Interface
    std::unique_ptr<Metavision::PeriodicFrameGenerationAlgorithm> event_frame_gen_; ///< Event Frame Generator
    std::unique_ptr<Metavision::TransposeEventsAlgorithm> transpose_events_filter_; ///< Transpose X/Y on events

    std::vector<Metavision::EventCD> buffer_filters_;

    std::string input_file_;
    bool vertical_jets_;
    std::uint32_t accumulation_time_us_;
};

bool Pipeline::parse_command_line(int argc, char *argv[]) {
    const std::string short_program_desc("Metavision Jet Monitoring Calibration Tool\n");
    const std::string long_program_desc(
        short_program_desc +
        "Make sure the nozzle is in the field of view of the camera and is firing either horizontally or vertically.\n"
        "Press 'Space' when the jet is clearly visible on the display of events.\n"
        "Once ROIs have been drawn on the display, press 'Enter' to print --detection-roi and --camera-roi in the "
        "console. "
        "Then, run the Jet Monitoring sample using these two command line arguments.\n\n"
        "Press 'Space' to play/pause events\n"
        "Press 'B' to define the baseline\n"
        "Press 'C' to define the Camera ROI\n"
        "Press 'J' to define the Jet ROI\n"
        "Press 'Enter' to print ROIs\n"
        "Press 'Q' or 'Escape' to exit\n");

    bpo::options_description options_desc("");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-file,i",        bpo::value<std::string>(&input_file_)->default_value(""), "Path to input file. If not specified, the camera live stream is used.")
        ("accumulation-time,a", bpo::value<std::uint32_t>(&accumulation_time_us_)->default_value(10000), "Accumulation time (in us) to use to generate a frame.")
        ("vertical-jets,v",     bpo::bool_switch(&vertical_jets_)->default_value(false), "Rotate the camera 90 degrees clockwise in case of a nozzle firing jets vertically in the FOV.")
        ;
    // clang-format on

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

    MV_LOG_INFO() << long_program_desc;

    return true;
}

bool Pipeline::init() {
    // Create camera
    if (input_file_ == "")
        camera_ = Metavision::Camera::from_first_available();
    else {
        try {
            camera_ = Metavision::Camera::from_file(input_file_);
        } catch (Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return false;
        }
    }

    const auto &geometry = camera_.geometry();
    int width, height;
    if (vertical_jets_) {
        transpose_events_filter_ = std::make_unique<Metavision::TransposeEventsAlgorithm>();
        width                    = geometry.height();
        height                   = geometry.width();
    } else {
        width  = geometry.width();
        height = geometry.height();
    }

    // GUI
    gui_ = std::make_unique<Metavision::JetMonitoringCalibrationGUI>(width, height, vertical_jets_);

    // Event Frame Generator
    event_frame_gen_ =
        std::make_unique<Metavision::PeriodicFrameGenerationAlgorithm>(width, height, accumulation_time_us_);
    event_frame_gen_->set_output_callback(
        [this](Metavision::timestamp ts, cv::Mat &cd_frame) { gui_->swap_cd_frame_if_required(cd_frame); });

    // Add camera callbacks
    camera_.add_runtime_error_callback([](const Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); });

    camera_.cd().add_callback([this](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        if (transpose_events_filter_) {
            transpose_events_filter_->process_events(begin, end, std::back_inserter(buffer_filters_));
            event_frame_gen_->process_events(buffer_filters_.cbegin(), buffer_filters_.cend());
        } else
            event_frame_gen_->process_events(begin, end);
    });

    // Stop the pipeline when the camera is stopped
    camera_.add_status_change_callback([this](const Metavision::CameraStatus &status) {
        if (status == Metavision::CameraStatus::STOPPED)
            if (input_file_ != "")
                MV_LOG_INFO() << "The file has been entirely processed. The app now uses the last saved cd frame.";
    });

    return true;
}

bool Pipeline::run() {
    if (!camera_.start()) {
        MV_LOG_ERROR() << "The camera could not be started.";
        return false;
    }

    // Wait until the closing of the window
    while (!gui_->should_close()) {
        Metavision::EventLoop::poll_and_dispatch();
        gui_->update();
    }

    // Stop the camera if it hasn't already been stopped
    try {
        camera_.stop();
    } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }

    return true;
}

int main(int argc, char **argv) {
    Pipeline pipeline;

    // Parse command line
    if (!pipeline.parse_command_line(argc, argv))
        return 1;

    if (!pipeline.init())
        return 2;

    if (!pipeline.run())
        return 3;
}