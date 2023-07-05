/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// This code sample demonstrates how to display the results of the sparse optical flow algorithm.
#include <iostream>
#include <functional>
#include <chrono>
#include <boost/program_options.hpp>

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/sdk/cv/algorithms/sparse_optical_flow_algorithm.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/hal/facilities/i_antiflicker_module.h>
#include <metavision/hal/facilities/i_event_trail_filter_module.h>
#include <metavision/hal/facilities/i_erc_module.h>
#include "sparse_flow_frame_generation.h"

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string bias_file_path;
    uint32_t afk_min_freq     = 90;
    uint32_t afk_max_freq     = 120;
    bool afk_band_pass        = true;
    uint32_t stc_threshold    = 10000;
    uint32_t erc_event_rate   = 20000000;
    bool hardware_stc_enabled = false;
    std::string in_file_path;
    std::string out_avi_file_path;
    bool benchmark               = false;
    bool no_display              = false;
    bool realtime_playback_speed = true;

    const std::string program_desc(
        "Code sample showing how to use Metavision SDK to display results of sparse optical flow.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("bias-file,b", po::value<std::string>(&bias_file_path), "Apply bias settings on the camera")
        ("min-freq,m", po::value<uint32_t>(&afk_min_freq), "AFK: Lowest frequency of the band (in Hz), for Gen4.1 sensors and newer")
        ("max-freq,M", po::value<uint32_t>(&afk_max_freq), "AFK: Highest frequency of the band (in Hz), for Gen4.1 sensors and newer")
        ("band-pass,s", po::value<bool>(&afk_band_pass), "AFK: True for band-pass (by default), and False for band-cut, for Gen4.1 sensors and newer")
        ("stc-threshold,t", po::value<uint32_t>(&stc_threshold), "STC: filtering threshold delay (in us), for Gen4.1 sensors and newer")
        ("erc-event-rate,e", po::value<uint32_t>(&erc_event_rate)->default_value(20000000), "ERC: ERC target event rate (in event/s), for Gen4 sensors and newer")
        ("input-file,i", po::value<std::string>(&in_file_path), "Path to input file. If not specified, the camera live stream is used.")
        ("output-avi-file,o", po::value<std::string>(&out_avi_file_path)->default_value(""), "Path to output AVI file.")
        ("benchmark", po::bool_switch(&benchmark), "Configure pipeline to enable timing of the sparse flow algorithm specifically")
        ("no-display,d", po::bool_switch(&no_display)->default_value(false), "Disable output display window")
        ("realtime-playback-speed", po::value<bool>(&realtime_playback_speed)->default_value(true), "Replay events at speed of recording if true, otherwise as fast as possible")
        ;
    // clang-format on

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return 1;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return 0;
    }

    const bool enable_display      = !no_display && !benchmark;
    const bool enable_video_writer = !out_avi_file_path.empty() && !benchmark;
    const bool enable_visu         = enable_display || enable_video_writer;

    // Initialize the camera
    Metavision::Camera camera;
    if (in_file_path.empty()) {
        try {
            camera = Metavision::Camera::from_first_available();
        } catch (const Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return 1;
        }
        // Configure biases:
        if (!bias_file_path.empty()) {
            camera.biases().set_from_file(bias_file_path);
        }
        // Configure AFK
        if (vm.count("min-freq") || vm.count("max-freq")) {
            try {
                auto *afk = camera.get_device().get_facility<Metavision::I_AntiFlickerModule>();
                auto mode = afk_band_pass ? Metavision::I_AntiFlickerModule::AntiFlickerMode::BAND_PASS :
                                            Metavision::I_AntiFlickerModule::AntiFlickerMode::BAND_STOP;

                afk->set_filtering_mode(mode);
                afk->set_frequency_band(afk_min_freq, afk_max_freq);
                afk->enable(true);
            } catch (Metavision::CameraException &) {
                MV_LOG_ERROR() << "AFK not supported on this camera (available on Gen4.1 sensors and newer).";
            }
        }
        // Configure STC
        if (vm.count("stc-threshold")) {
            try {
                auto *event_trail_filter = camera.get_device().get_facility<Metavision::I_EventTrailFilterModule>();
                event_trail_filter->set_type(Metavision::I_EventTrailFilterModule::Type::STC_CUT_TRAIL);
                event_trail_filter->set_threshold(stc_threshold);
                event_trail_filter->enable(true);
                hardware_stc_enabled = true;
            } catch (Metavision::CameraException &) {
                MV_LOG_ERROR() << "STC not supported on this camera (available on Gen4.1 sensors and newer).";
            }
        }
        // Configure ERC
        try {
            auto *erc = camera.get_device().get_facility<Metavision::I_ErcModule>();
            erc->set_cd_event_rate(erc_event_rate);
            erc->enable(true);
        } catch (Metavision::CameraException &) {
            MV_LOG_ERROR() << "ERC not supported on this camera (available on Gen4 sensors and newer).";
        }

    } else {
        camera = Metavision::Camera::from_file(
            in_file_path, Metavision::FileConfigHints().real_time_playback(realtime_playback_speed && !benchmark));
    }

    const unsigned short width  = camera.geometry().width();
    const unsigned short height = camera.geometry().height();

    /// Data flow:
    //
    //  0 (Cam) -->-- 1 (STC) ----------->-----------  2 (Flow)
    //                |                                |
    //                v                                v
    //                |                                |
    //                |------>------->-<--------<------|
    //                                |
    //                                v
    //                                |
    //                                3 (Flow Frame Generator)
    //                                |
    //                                v
    //                                |
    //                |------<-------<->-------->------|
    //                |                                |
    //                v                                v
    //                |                                |
    //                4 (Video writer)                 5 (Display)
    //

    Metavision::SpatioTemporalContrastAlgorithm stc_algo(width, height, 0);
    stc_algo.set_threshold(stc_threshold);

    Metavision::SparseOpticalFlowAlgorithm flow_algo(width, height,
                                                     Metavision::SparseOpticalFlowConfig::Preset::FastObjects);

    std::unique_ptr<Metavision::SparseFlowFrameGeneration> flow_frame_gen;
    if (enable_visu) {
        flow_frame_gen = std::make_unique<Metavision::SparseFlowFrameGeneration>(width, height, 30);
    }

    // Vector of CD events and vector of OpticalFlow events to store the output of the two algorithms
    Metavision::timestamp ts_first = -1, ts_last = -1;
    std::vector<Metavision::EventCD> stc_output;
    std::vector<Metavision::EventOpticalFlow> flow_output;
    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        if (begin != end) {
            if (ts_first == -1)
                ts_first = begin->t;
            ts_last = std::prev(end)->t;
        }
        if (hardware_stc_enabled) {
            // When sensor's STC is available, we don't use software STC
            stc_output.insert(stc_output.end(), begin, end);
        } else {
            stc_algo.process_events(begin, end, std::back_inserter(stc_output));
        }
        flow_algo.process_events(stc_output.begin(), stc_output.end(), std::back_inserter(flow_output));

        // Call the frame generator on the processed events
        if (flow_frame_gen) {
            flow_frame_gen->process_cd_events(stc_output);
            flow_frame_gen->process_flow_events(flow_output);
        }
        stc_output.clear();
        flow_output.clear();
    });

    //  Create a video from the generated frames
    std::unique_ptr<cv::VideoWriter> video_writer;
    if (enable_video_writer) {
        video_writer = std::make_unique<cv::VideoWriter>(out_avi_file_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                                         60, cv::Size(width, height));
    }

    // Initialize the window if not disabled
    std::unique_ptr<Metavision::Window> window;
    if (enable_display) {
        window = std::make_unique<Metavision::Window>("Sparse Optical Flow", width, height,
                                                      Metavision::BaseWindow::RenderMode::BGR);
        window->set_keyboard_callback(
            [&window](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
                if (action == Metavision::UIAction::RELEASE &&
                    (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                    window->set_close_flag();
                }
            });
    }

    if (flow_frame_gen) {
        flow_frame_gen->set_output_callback([&](Metavision::timestamp, const cv::Mat &frame) {
            if (window) {
                window->show(frame);
            }
            if (video_writer) {
                video_writer->write(frame);
            }
        });
    }

    const auto start = std::chrono::high_resolution_clock::now();
    camera.start();
    while (camera.is_running()) {
        Metavision::EventLoop::poll_and_dispatch(20);
        if (window && window->should_close()) {
            break;
        }
    }
    camera.stop();
    const auto end     = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    MV_LOG_INFO() << "Ran in" << Metavision::Log::no_space << static_cast<float>(elapsed.count()) / 1000.f << "s";
    MV_LOG_INFO() << "Record duration " << Metavision::Log::no_space << static_cast<float>(ts_last - ts_first) / 1e6f
                  << "s";
    if (video_writer) {
        MV_LOG_INFO() << "Wrote video file:" << out_avi_file_path;
    }

    return 0;
}
