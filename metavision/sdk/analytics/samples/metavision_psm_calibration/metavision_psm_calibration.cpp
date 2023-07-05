/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/log.h>

#include "roi_line_events_loader.h"
#include "psm_display_simulator.h"

using timestamp   = Metavision::timestamp;
using SimuRunType = Metavision::PsmDisplaySimulator::RunType;

namespace bpo = boost::program_options;

class Pipeline {
public:
    Pipeline() = default;

    ~Pipeline() = default;

    /// @brief Parses command line attributes
    bool parse_command_line(int argc, char *argv[]);

    /// @brief Initializes pipeline
    bool init();

    /// @brief Runs pipeline
    void run();

private:
    void run_simulator(SimuRunType run_type);

    void run_simulator_at_trackbar_ts();

    /// @brief Handles key-events
    void key_events_handler(int key);

    // Lines of interest
    int min_y_line_;        ///< Min y to place the first line
    int max_y_line_;        ///< Max y to place the last line
    int num_lines_;         ///< Number of lines for processing between min-y and max-y
    std::vector<int> rows_; ///< Vector containing the ordinates of the lines of interest

    // Events loading and preprocessing
    std::unique_ptr<Metavision::RoiLineEventsLoader> events_loader_;
    std::string input_raw_path_;
    timestamp process_from_;      ///< Start time to process events
    timestamp process_to_;        ///< End time to process events
    short polarity_;              ///< Process only events of this polarity
    timestamp activity_time_ths_; ///< Length of the time window for activity filtering (in us)
    bool transpose_axis_;         ///< Set to true to rotate the camera 90 degrees clockwise in case of particles moving
                                  ///< horizontally in FOV

    // Display window
    int width_;
    int height_;
    bool display_           = true;
    bool run_mode_          = false;
    bool show_help_         = false;
    bool help_is_displayed_ = false;

    // Help overlay images
    cv::Mat help_overlay_run_;
    cv::Mat help_overlay_pause_;

    // Opencv trackbar
    int trackbar_cb_elapsed_ts_100us_;
    timestamp min_events_ts_;
    timestamp max_events_ts_;

    // PSM simulator
    std::unique_ptr<Metavision::PsmDisplaySimulator> psm_simulator_;
    Metavision::PsmDisplaySimulatorConfig psm_simu_config_;
};

bool Pipeline::parse_command_line(int argc, char *argv[]) {
    const std::string short_program_desc("Metavision Particle Size Measurement Calibration Tool\n");

    const std::string long_program_desc(
        short_program_desc +
        "A CSV configuration file will be generated with PsmAlgorithm parameters. "
        "Modify it to refresh dynamically the config used by the algorithm.\n"
        "\n"
        "A window displays the result of the PsmAlgorithm. Horizontal red lines correspond to the "
        "selected lines of interest. As for the horizontal yellow segments, they reflect what the "
        "algorithm sees along the line once it has processed the events. When a particle has been matched over several "
        "rows, its 2D reconstruction appears in green and its trajectory in red. The number below the reconstructed "
        "contour is the particle size estimate.\n"
        "\n"
        "The player allows to go back and forth in time, and even to jump to specific timestamps using the trackbar.\n"
        "\n"
        "Press 'space' to play/pause the simulation.\n"
        "Press 'h' to see available options.\n");

    bpo::options_description options_desc;
    bpo::options_description base_options("Base options");
    // clang-format off
    base_options.add_options()
        ("help,h", "Produce help message.")
        ("input-raw-file,i",   bpo::value<std::string>(&input_raw_path_)->required(), "Path to input RAW file. If not specified, the camera live stream is used.")
        ("output-directory,o", bpo::value<std::string>(&psm_simu_config_.output_dir)->default_value("/tmp"), "Output directory used to save the CSV config file.")
        ("process-from,s",     bpo::value<timestamp>(&process_from_)->default_value(0), "Start time to process events (in us).")
        ("process-to,e",       bpo::value<timestamp>(&process_to_)->default_value(-1), "End time to process events (in us).")
        ;
    // clang-format on

    bpo::options_description filter_options("Filtering options");
    // clang-format off
    filter_options.add_options()
        ("activity-ths", bpo::value<timestamp>(&activity_time_ths_)->default_value(0), "Length of the time window for activity filtering (in us).")
        ("polarity",     bpo::value<short>(&polarity_)->default_value(0), "Which event polarity to process : 0 (OFF), 1 (ON), -1 (ON & OFF). By default it uses only OFF events.")
        ("rotate,r",     bpo::bool_switch(&transpose_axis_)->default_value(false), "Rotate the camera 90 degrees clockwise in case of particles moving horizontally in FOV.")
        ;
    // clang-format on

    bpo::options_description lines_options("Lines options");
    // clang-format off
    lines_options.add_options()
        ("min-y",               bpo::value<int>(&min_y_line_)->default_value(200), "Min y to place the first line cluster tracker.")
        ("max-y",               bpo::value<int>(&max_y_line_)->default_value(300), "Max y to place the last line cluster tracker.")
        ("num-lines,n",         bpo::value<int>(&num_lines_)->default_value(6), "Number of lines for processing between min-y and max-y.")
        ("objects-moving-up,u", bpo::bool_switch(&psm_simu_config_.is_going_up)->default_value(false), "Specify if the particles are going upwards.")
        ("persistence-contour", bpo::value<int>(&psm_simu_config_.persistence_contour)->default_value(40), "Once a particle contour has been estimated, keep the drawing superimposed on the display for a given number of frames.")
     ;
    // clang-format on

    bpo::options_description particle_options("Particle options");
    // clang-format off
    particle_options.add_options()
        ("avg-speed", bpo::value<int>(&psm_simu_config_.avg_speed_pix_ms)->default_value(50), "Average particle speed in pix/ms.")
        ("avg-size",  bpo::value<int>(&psm_simu_config_.avg_size_pix)->default_value(20), "Approximate particle size in pix.")
        ;
    // clang-format on

    options_desc.add(base_options).add(filter_options).add(lines_options).add(particle_options);

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

    // Check polarity
    if (polarity_ != 1 && polarity_ != 0 && polarity_ != -1) {
        MV_LOG_ERROR() << "The polarity is not valid:" << polarity_;
        return false;
    }

    // Check positions of the line counters
    if (max_y_line_ <= min_y_line_) {
        MV_LOG_ERROR() << Metavision::Log::no_space << "The range of y-positions for the line counters is not valid : "
                       << "[" << min_y_line_ << ", " << max_y_line_ << "].";
        return false;
    }

    MV_LOG_INFO() << long_program_desc;

    return true;
}

bool Pipeline::init() {
    // Define lines of interest
    rows_.reserve(num_lines_);
    const int y_line_step = (max_y_line_ - min_y_line_) / (num_lines_ - 1);
    for (int i = 0; i < num_lines_; ++i) {
        const int line_ordinate = min_y_line_ + y_line_step * i;
        rows_.push_back(line_ordinate);
    }

    // Load events on the lines of interest
    MV_LOG_INFO() << "Loading events...";
    std::vector<Metavision::EventCD> events_tmp; // Temporary vector that will be moved
    events_loader_ = std::make_unique<Metavision::RoiLineEventsLoader>(rows_);
    cv::Size camera_size;
    if (!events_loader_->load_events_from_rawfile(input_raw_path_, events_tmp, camera_size, transpose_axis_, polarity_,
                                                  activity_time_ths_, process_from_, process_to_)) {
        MV_LOG_ERROR() << "Failed to load events from RAW file:" << input_raw_path_;
        return false;
    }
    MV_LOG_INFO() << "Events have been successfully loaded.";

    width_  = camera_size.width;
    height_ = camera_size.height;

    min_events_ts_ = events_tmp.front().t;
    max_events_ts_ = events_tmp.back().t;
    MV_LOG_INFO() << Metavision::Log::no_space << (max_events_ts_ - min_events_ts_) / 1000
                  << "ms of events have been loaded on the lines of interest. Time interval: "
                  << "[" << min_events_ts_ << ", " << max_events_ts_ << "]";

    if ((max_events_ts_ - min_events_ts_) / 100 == 0) {
        MV_LOG_ERROR() << "Too few events. Duration must be greater than 100us.";
        return false;
    }

    try {
        psm_simulator_ = std::make_unique<Metavision::PsmDisplaySimulator>(width_, height_, psm_simu_config_, rows_,
                                                                           std::move(events_tmp));
    } catch (const std::invalid_argument &ia) {
        MV_LOG_ERROR() << "Failed to construct PsmDisplaySimulator:";
        MV_LOG_ERROR() << ia.what();
        return false;
    }

    // Initialized algorithm display
    run_simulator(SimuRunType::BEGIN);

    // Initialize help overlay in RUN mode
    const cv::Vec3b text_color(0, 255, 255);
    help_overlay_run_.create(height_, width_, CV_8UC3);
    help_overlay_run_.setTo(cv::Vec3b(0, 0, 0));
    cv::putText(help_overlay_run_, "KeyEvents:", {40, 60}, cv::FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv::LINE_8);
    cv::putText(help_overlay_run_, "Press Q or Escape to quit the application", {40, 100}, cv::FONT_HERSHEY_SIMPLEX,
                0.8, text_color, 1, cv::LINE_8);
    cv::putText(help_overlay_run_, "Press H to hide/show help", {40, 130}, cv::FONT_HERSHEY_SIMPLEX, 0.8, text_color, 1,
                cv::LINE_8);
    cv::putText(help_overlay_run_, "Press B to go to the Begin", {40, 160}, cv::FONT_HERSHEY_SIMPLEX, 0.8, text_color,
                1, cv::LINE_8);
    cv::putText(help_overlay_run_, "Press E to go to the End", {40, 190}, cv::FONT_HERSHEY_SIMPLEX, 0.8, text_color, 1,
                cv::LINE_8);
    cv::putText(help_overlay_run_, "Press Space to play/pause", {40, 220}, cv::FONT_HERSHEY_SIMPLEX, 0.8, text_color, 1,
                cv::LINE_8);

    // Initialize help overlay in PAUSE mode
    help_overlay_run_.copyTo(help_overlay_pause_);
    cv::putText(help_overlay_pause_, "---", {40, 250}, cv::FONT_HERSHEY_SIMPLEX, 0.8, text_color, 1, cv::LINE_8);
    cv::putText(help_overlay_pause_, "Press R to Reset the configuration file", {40, 280}, cv::FONT_HERSHEY_SIMPLEX,
                0.8, text_color, 1, cv::LINE_8);
    cv::putText(help_overlay_pause_, "Press Z to move to previous frame", {40, 310}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                text_color, 1, cv::LINE_8);
    cv::putText(help_overlay_pause_, "Press Y to move to next frame", {40, 340}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                text_color, 1, cv::LINE_8);

    return true;
}

void Pipeline::run_simulator(SimuRunType run_type) {
    psm_simulator_->run(run_type);
    trackbar_cb_elapsed_ts_100us_ = static_cast<int>(psm_simulator_->get_crt_ts() - min_events_ts_) / 100;
}

void Pipeline::run_simulator_at_trackbar_ts() {
    psm_simulator_->run_at_given_ts(min_events_ts_ + 100 * trackbar_cb_elapsed_ts_100us_);
}

void Pipeline::key_events_handler(int key) {
    switch (key) {
    case 'q':
    case 27: // Escape
        display_ = false;
        break;
    case 32: // Space
        run_mode_          = !run_mode_;
        help_is_displayed_ = false;
        if (run_mode_)
            run_simulator_at_trackbar_ts();
        else
            run_simulator(SimuRunType::RERUN);
        break;
    case 'h':
        help_is_displayed_ = false;
        show_help_         = !show_help_;
        if (!run_mode_ && !show_help_)
            run_simulator(SimuRunType::RERUN);
        break;
    case 'e':
        run_simulator(SimuRunType::END);
        break;
    case 'b':
        run_simulator(SimuRunType::BEGIN);
        break;
    default:
        break;
    }

    if (!run_mode_) {
        switch (key) {
        case 'r':
            MV_LOG_INFO() << "Reset configuration file and update display";
            psm_simulator_->reset_config();
            break;
        case 'z':
            run_simulator(SimuRunType::BACKWARD);
            break;
        case 'y':
            run_simulator(SimuRunType::FORWARD);
            break;
        default:
            break;
        }
    }
}

void Pipeline::run() {
    display_           = true;
    run_mode_          = false;
    show_help_         = false;
    help_is_displayed_ = false;

    const std::string main_window_name = "Particle Size Measurement - Calibration";
    cv::namedWindow(main_window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(main_window_name, width_, height_);
    cv::moveWindow(main_window_name, 0, 0);

    cv::Mat front_img_;
    while (display_) {
        psm_simulator_->swap_back_image_if_updated(front_img_);

        if (show_help_) {
            if (run_mode_ || !help_is_displayed_) {
                help_is_displayed_ = true;
                cv::addWeighted(front_img_, 0.5, (run_mode_ ? help_overlay_run_ : help_overlay_pause_), 0.5, 0,
                                front_img_);
            }
        }

        cv::createTrackbar("Elapsed ts (in 100us)", main_window_name, &trackbar_cb_elapsed_ts_100us_,
                           static_cast<int>(max_events_ts_ - min_events_ts_) / 100);
        cv::namedWindow(main_window_name, cv::WINDOW_NORMAL);
        cv::imshow(main_window_name, front_img_);
        const int key = cv::waitKey(1) & 0xff; // When no key is pressed, some versions of
                                               // opencv returns -1, some 255. To catch both,
                                               // we do & 0xff and then compare to 255

        if (key != 255)
            key_events_handler(key);

        if (run_mode_)
            run_simulator(SimuRunType::FORWARD);

        if (psm_simulator_->is_at_the_end()) {
            run_mode_          = false;
            help_is_displayed_ = false;
        }
    }
}

/// Main function
int main(int argc, char *argv[]) {
    Pipeline pipeline;

    // Parse command line
    if (!pipeline.parse_command_line(argc, argv))
        return 1;

    if (!pipeline.init())
        return 2;

    pipeline.run();

    return 0;
}
