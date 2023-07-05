/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// Tool for generating and recording a blinking pattern

#include <functional>
#include <regex>
#include <chrono>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/pipeline/pipeline.h>
#include <metavision/sdk/core/pipeline/frame_composition_stage.h>
#include <metavision/sdk/core/pipeline/frame_generation_stage.h>
#include <metavision/sdk/driver/pipeline/camera_stage.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/sdk/calibration/algorithms/blinking_frame_generator_algorithm.h>
#include <metavision/sdk/calibration/utils/pattern_blinker.h>
#include <metavision/sdk/calibration/configs/dft_high_freq_scorer_algorithm_config.h>
#include <metavision/sdk/calibration/algorithms/dft_high_freq_scorer_algorithm.h>
#include <metavision/sdk/calibration/pipeline/blinking_pattern_generator_stage.h>
#include <metavision/sdk/calibration/pipeline/pattern_blinker_stage.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/pipeline/frame_display_stage.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

// Application's parameters
struct Config {
    // Input/Output parameters
    std::string raw_file_path_;
    std::string output_dir_path_ = "";

    // Blinking frame generator algorithm's parameters
    Metavision::timestamp accumulation_time_;
    int min_num_blinking_pixels_;
    float blinking_pixels_ratios_on_;
    float blinking_pixels_ratios_off_;
    bool enable_event_count_;

    // Pattern Blinker's parameters
    std::string pattern_image_path_;
    int pattern_blinker_height_;
    Metavision::timestamp pattern_blinker_refresh_period_us_;
};

bool get_pipeline_configuration(int argc, char *argv[], Config &config) {
    const std::string program_desc(
        "Tool for camera focusing by means of a blinking pattern, using Metavision Calibration SDK.\n");

    po::options_description options_desc;
    po::options_description base_options("Base options");
    // clang-format off
    base_options.add_options()
        ("help,h", "Produce help message.")
        ("input-raw-file,i", po::value<std::string>(&config.raw_file_path_), "Path to input RAW file. If not specified, the camera live stream is used.")
        ("output-dir,o",          po::value<std::string>(&config.output_dir_path_)->default_value(""), "Path to the folder where the 2D detections will be saved. If the folder does not exists, it will be created.")
        ;
    // clang-format on

    po::options_description blinking_frame_generator_options("Blinking Frame Generator options");
    // clang-format off
    blinking_frame_generator_options.add_options()
        ("accumulation-time,a", po::value<Metavision::timestamp>(&config.accumulation_time_)->default_value(2e5), "Window of time during which events are considered to detect if a pixel is blinking.")
        ("min-blink-pix,m", po::value<int>(&config.min_num_blinking_pixels_)->default_value(0), "Minimum number of pixels needed to be detected before outputting a frame.")
        ("ratio-on",        po::value<float>(&config.blinking_pixels_ratios_on_)->default_value(1.0f), "The acceptable ratio of pixels that received only positive events over the number of pixels that received both during the accumulation window.")
        ("ratio-off",       po::value<float>(&config.blinking_pixels_ratios_off_)->default_value(1.0f),  "The acceptable ratio of pixels that received only negative events over the number of pixels that received both during the accumulation window.")
        ("event-count",       po::value<bool>(&config.enable_event_count_)->default_value(false),  "Accumulate events to a grayscale image instead of creating a binary output.")
        ;
    // clang-format on

    po::options_description pattern_blinker_options("Pattern Blinker options");
    // clang-format off
    pattern_blinker_options.add_options()
        ("pattern-image-path",  po::value<std::string>(&config.pattern_image_path_), "If a path to a pattern file is provided, display a blinking pattern on screen")
        ("pattern-blinker-height",  po::value<int>(&config.pattern_blinker_height_)->default_value(1080), "Height of the blinking pattern.")
        ("pattern-blinker-refresh", po::value<Metavision::timestamp>(&config.pattern_blinker_refresh_period_us_)->default_value(1e4), "Refresh period of the pattern blinker in us.")
        ;
    // clang-format on

    options_desc.add(base_options).add(blinking_frame_generator_options).add(pattern_blinker_options);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return false;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return false;
    }

    if (config.output_dir_path_ != "" && !fs::exists(config.output_dir_path_)) {
        try {
            fs::create_directories(config.output_dir_path_);
        } catch (fs::filesystem_error &e) {
            MV_LOG_ERROR() << "Unable to create folder" << config.output_dir_path_;
            return false;
        }
    }

    return true;
}

int main(int argc, char *argv[]) {
    Config conf_;

    if (!get_pipeline_configuration(argc, argv, conf_))
        return 1;

    const auto start = std::chrono::high_resolution_clock::now();

    Metavision::Pipeline p(true);

    Metavision::Camera camera;
    if (conf_.raw_file_path_.empty()) {
        try {
            camera = Metavision::Camera::from_first_available();
        } catch (const Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return 1;
        }
    } else {
        camera = Metavision::Camera::from_file(conf_.raw_file_path_,
                                               Metavision::FileConfigHints().real_time_playback(false));
    }

    const unsigned short width  = camera.geometry().width();
    const unsigned short height = camera.geometry().height();

    const Metavision::timestamp event_buffer_duration_ms = 100;
    const int display_fps                                = 10;

    // 0) Camera stage
    auto &cam_stage =
        p.add_stage(std::make_unique<Metavision::CameraStage>(std::move(camera), event_buffer_duration_ms));

    // 1) Events frame stage
    auto &events_frame_stage = p.add_stage(
        std::make_unique<Metavision::FrameGenerationStage>(width, height, event_buffer_duration_ms, display_fps),
        cam_stage);
    // 2) Blinking Frame Generator stage
    Metavision::BlinkingFrameGeneratorAlgorithmConfig blinking_config(
        conf_.accumulation_time_, conf_.min_num_blinking_pixels_, conf_.blinking_pixels_ratios_on_,
        conf_.blinking_pixels_ratios_off_, -1, conf_.enable_event_count_);
    auto &blinking_frame_generator_stage = p.add_stage(std::make_unique<Metavision::BlinkingFrameGeneratorStage>(
                                                           width, height, blinking_config, conf_.output_dir_path_),
                                                       cam_stage);

    // 3) Frame composer stage
    /// [CONNECT_FRAME_COMPOSER_BEGIN]
    auto &frame_composer_stage = p.add_stage(std::make_unique<Metavision::FrameCompositionStage>(display_fps, 0));
    frame_composer_stage.add_previous_frame_stage(events_frame_stage, 0, 0, width, height);
    frame_composer_stage.add_previous_frame_stage(blinking_frame_generator_stage, width, 0, width, height);

    //// 4) Stage displaying the raw events and the blinking events
    const auto composed_width  = frame_composer_stage.frame_composer().get_total_width();
    const auto composed_height = frame_composer_stage.frame_composer().get_total_height();
    auto &display_stage =
        p.add_stage(std::make_unique<Metavision::FrameDisplayStage>(
                        "Raw events, Blinking events and High Frequency score ", composed_width, composed_height),
                    frame_composer_stage);
    /// [CONNECT_FRAME_COMPOSER_END]

    if (!conf_.pattern_image_path_.empty()) {
        try {
            // 7) Pattern Blinker Stage
            auto &pattern_blinker_stage = p.add_stage(std::make_unique<Metavision::PatternBlinkerStage>(
                conf_.pattern_image_path_, conf_.pattern_blinker_height_, conf_.pattern_blinker_refresh_period_us_));

            // 8) Stage displaying the blinking pattern
            const auto &blinking_pattern_size = pattern_blinker_stage.get_display_size();
            auto &disp_blinker_stage =
                p.add_stage(std::make_unique<Metavision::FrameDisplayStage>(
                                "Blinking Pattern", blinking_pattern_size.width, blinking_pattern_size.height,
                                Metavision::Window::RenderMode::GRAY),
                            pattern_blinker_stage);
        } catch (const std::runtime_error &e) {
            MV_LOG_ERROR() << e.what();
            return 1;
        }
    }

    // Run the pipeline and wait for its completion
    p.run();

    const auto end     = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    MV_LOG_INFO() << "Ran in" << static_cast<float>(elapsed.count()) / 1000.f << "s";

    return 0;
}
