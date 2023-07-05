/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// This code sample demonstrates how to create a pipeline displaying the results of dense optical flow algorithms.
#include <iostream>
#include <chrono>
#include <functional>
#include <unordered_map>
#include <boost/program_options.hpp>
#include <metavision/sdk/core/algorithms/event_buffer_reslicer_algorithm.h>
#include <metavision/sdk/core/algorithms/on_demand_frame_generation_algorithm.h>
#include <metavision/sdk/cv/algorithms/dense_flow_frame_generator_algorithm.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/sdk/cv/algorithms/plane_fitting_flow_algorithm.h>
#include <metavision/sdk/cv/algorithms/triplet_matching_flow_algorithm.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/utils/window.h>

namespace po = boost::program_options;

enum class DenseFlowType { PlaneFitting, TripletMatching };

const std::unordered_map<DenseFlowType, std::string> kDenseFlowTypeToStr = {
    {DenseFlowType::PlaneFitting, "PlaneFitting"}, {DenseFlowType::TripletMatching, "TripletMatching"}};
const std::unordered_map<std::string, DenseFlowType> kStrToDenseFlowType = {
    {"PlaneFitting", DenseFlowType::PlaneFitting}, {"TripletMatching", DenseFlowType::TripletMatching}};

std::istream &operator>>(std::istream &is, DenseFlowType &type) {
    std::string s;
    is >> s;
    auto it = kStrToDenseFlowType.find(s);
    if (it == kStrToDenseFlowType.cend())
        throw std::runtime_error("Failed to convert string to DenseFlowType");
    type = it->second;
    return is;
}

std::ostream &operator<<(std::ostream &os, const DenseFlowType &type) {
    auto it = kDenseFlowTypeToStr.find(type);
    if (it == kDenseFlowTypeToStr.cend())
        throw std::runtime_error("Failed to convert DenseFlowType to string");
    os << it->second;
    return os;
}

struct UserArguments {
    std::string bias_file_path;
    uint32_t afk_min_freq   = 90;
    uint32_t afk_max_freq   = 120;
    bool afk_band_pass      = true;
    uint32_t stc_threshold  = 10000;
    uint32_t erc_event_rate = 20000000;
    std::string in_file_path;
    std::string out_avi_file_path;
    DenseFlowType flow_type;
    float receptive_field_radius;
    float min_flow_mag, max_flow_mag;
    bool benchmark  = false;
    bool no_display = false;
    Metavision::timestamp processing_period;
    float visualization_flow_scale;
};

void configure_live_camera(Metavision::Camera &camera, const po::variables_map &vm, const UserArguments &args,
                           bool &hardware_stc_enabled) {
    // Configure biases:
    if (!args.bias_file_path.empty()) {
        camera.biases().set_from_file(args.bias_file_path);
    }
    // Configure AFK
    if (vm.count("min-freq") || vm.count("max-freq")) {
        try {
            auto *afk = camera.get_device().get_facility<Metavision::I_AntiFlickerModule>();
            auto mode = args.afk_band_pass ? Metavision::I_AntiFlickerModule::AntiFlickerMode::BAND_PASS :
                                             Metavision::I_AntiFlickerModule::AntiFlickerMode::BAND_STOP;

            afk->set_filtering_mode(mode);
            afk->set_frequency_band(args.afk_min_freq, args.afk_max_freq);
            afk->enable(true);
        } catch (Metavision::CameraException &e) {
            MV_LOG_ERROR() << "AFK not supported on this camera (available on Gen4.1 sensors and newer).";
        }
    }
    // Configure STC
    if (vm.count("stc-threshold")) {
        try {
            auto *event_trail_filter = camera.get_device().get_facility<Metavision::I_EventTrailFilterModule>();
            event_trail_filter->set_type(Metavision::I_EventTrailFilterModule::Type::STC_CUT_TRAIL);
            event_trail_filter->set_threshold(args.stc_threshold);
            event_trail_filter->enable(true);
            hardware_stc_enabled = true;
        } catch (Metavision::CameraException &e) {
            MV_LOG_ERROR() << "STC not supported on this camera (available on Gen4.1 sensors and newer).";
        }
    }
    // Configure ERC
    try {
        auto *erc = camera.get_device().get_facility<Metavision::I_ErcModule>();
        erc->set_cd_event_rate(args.erc_event_rate);
        erc->enable(true);
    } catch (Metavision::CameraException &e) {
        MV_LOG_ERROR() << "ERC not supported on this camera (available on Gen4 sensors and newer).";
    }
}

int main(int argc, char *argv[]) {
    bool hardware_stc_enabled = false;
    UserArguments args;

    const std::string program_desc(
        "Code sample showing how to use Metavision SDK to display results of dense optical flow algorithms.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("bias-file,b", po::value<std::string>(&args.bias_file_path), "Apply bias settings on the camera")
        ("min-freq,m", po::value<uint32_t>(&args.afk_min_freq), "AFK: Lowest frequency of the band (in Hz), for Gen4.1 sensors and newer")
        ("max-freq,M", po::value<uint32_t>(&args.afk_max_freq), "AFK: Highest frequency of the band (in Hz), for Gen4.1 sensors and newer")
        ("band-pass,s", po::value<bool>(&args.afk_band_pass), "AFK: True for band-pass (by default), and False for band-cut, for Gen4.1 sensors and newer")
        ("stc-threshold,t", po::value<uint32_t>(&args.stc_threshold), "STC: filtering threshold delay (in us), for Gen4.1 sensors and newer")
        ("erc-event-rate,e", po::value<uint32_t>(&args.erc_event_rate)->default_value(20000000), "ERC: ERC target event rate (in event/s), for Gen4 sensors and newer")
        ("input-file,i", po::value<std::string>(&args.in_file_path)->default_value(""), "Path to input file. If empty, will try to open the first available live camera.")
        ("output-avi-file,o", po::value<std::string>(&args.out_avi_file_path)->default_value(""), "Path to output AVI file.")
        ("flow-type", po::value<DenseFlowType>(&args.flow_type)->default_value(DenseFlowType::TripletMatching), "Chosen type of dense flow algorithm to run, in {PlaneFitting, TripletMatching}")
        ("receptive-field-radius,r", po::value<float>(&args.receptive_field_radius)->default_value(3), "Radius of the receptive field, in pixels, used for flow estimation and converted for each method into the relevant radius.")
        ("min-flow", po::value<float>(&args.min_flow_mag)->default_value(10), "Minimum observable flow magnitude, in px/s.")
        ("max-flow", po::value<float>(&args.max_flow_mag)->default_value(1000), "Maximum observable flow magnitude, in px/s.")
        ("benchmark", po::bool_switch(&args.benchmark), "Configure pipeline to skip costly visualizations to enable timing of the dense flow algorithm specifically")
        ("no-display,d", po::bool_switch(&args.no_display), "Disable output display window")
        ("processing-period", po::value<Metavision::timestamp>(&args.processing_period)->default_value(33333), "Period for slicing and processing events and for generating flow visualization, in us.")
        ("visu-scale", po::value<float>(&args.visualization_flow_scale)->default_value(-1), "Flow magnitude used to scale the upper bound of the flow visualization, in px/s. If negative, will use 20\% of maximum flow magnitude.")
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

    if (args.visualization_flow_scale == -1)
        args.visualization_flow_scale = args.max_flow_mag / 5;

    const bool enable_display      = !args.no_display && !args.benchmark;
    const bool enable_video_writer = !args.out_avi_file_path.empty() && !args.benchmark;
    const bool enable_visu         = enable_display || enable_video_writer;

    // Initialize the camera
    Metavision::Camera camera;
    if (args.in_file_path.empty()) {
        try {
            camera = Metavision::Camera::from_first_available();
        } catch (const Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return 1;
        }
        configure_live_camera(camera, vm, args, hardware_stc_enabled);
    } else {
        try {
            camera = Metavision::Camera::from_file(args.in_file_path,
                                                   Metavision::FileConfigHints().real_time_playback(!args.benchmark));
        } catch (Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return 1;
        }
    }
    const unsigned short width  = camera.geometry().width();
    const unsigned short height = camera.geometry().height();

    /// Data flow:
    //
    //  0 (Cam) -->-- 1 (buffer reslicer) -->-- 2 (STC) ----------->-----------  3 (Flow)
    //                                          |                                |
    //                                          v                                v
    //                                          |                                |
    //                                          4 (CD Frame Generator)           5 (Flow Frame Generator)
    //                                          |                                |
    //                                          v                                v
    //                                          |------>------->-<--------<------|
    //                                                          |
    //                                                          v
    //                                                          |
    //                                          |------<-------<->-------->------|
    //                                          |                                |
    //                                          v                                v
    //                                          |                                |
    //                                          6 (Display)                      7 (Video writer)
    //

    std::unique_ptr<Metavision::SpatioTemporalContrastAlgorithm> stc_algo;
    if (!hardware_stc_enabled) {
        stc_algo = std::make_unique<Metavision::SpatioTemporalContrastAlgorithm>(width, height, 0);
        stc_algo->set_threshold(args.stc_threshold);
        MV_LOG_INFO() << "Instantiating software SpatioTemporalContrastAlgorithm with thresh=" << args.stc_threshold;
    }

    // Instantiate the flow estimation algorithms
    // The input receptive field radius represents the total area of the neighborhood that is used to estimate flow. We
    // use an algorithm-dependent heuristic to convert this into the search radius to be used for each algorithm.
    std::unique_ptr<Metavision::PlaneFittingFlowAlgorithm> plane_fitting_flow_algo;
    std::unique_ptr<Metavision::TripletMatchingFlowAlgorithm> triplet_matching_flow_algo;
    switch (args.flow_type) {
    case DenseFlowType::PlaneFitting: {
        const int radius = cvFloor(args.receptive_field_radius);
        MV_LOG_INFO() << "Instantiating PlaneFittingFlowAlgorithm with radius=" << radius;
        plane_fitting_flow_algo = std::make_unique<Metavision::PlaneFittingFlowAlgorithm>(width, height, radius, -1);
        break;
    }
    case DenseFlowType::TripletMatching: {
        const float radius = 0.5f * args.receptive_field_radius;
        MV_LOG_INFO() << "Instantiating TripletMatchingFlowAlgorithm with radius=" << radius;
        Metavision::TripletMatchingFlowAlgorithmConfig triplet_matching_config(radius, args.min_flow_mag,
                                                                               args.max_flow_mag);
        triplet_matching_flow_algo =
            std::make_unique<Metavision::TripletMatchingFlowAlgorithm>(width, height, triplet_matching_config);
        break;
    }
    default:
        throw std::runtime_error("Selected DenseFlowType is not implemented!");
    }

    //  Create a video from the generated frames
    std::unique_ptr<cv::VideoWriter> video_writer;
    if (enable_video_writer) {
        video_writer = std::make_unique<cv::VideoWriter>(
            args.out_avi_file_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 60, cv::Size(width, height));
    }

    // Initialize the window if not disabled
    std::unique_ptr<Metavision::Window> window_visu, window_legend;
    if (enable_display) {
        window_visu = std::make_unique<Metavision::Window>("Dense Optical Flow", width, height,
                                                           Metavision::BaseWindow::RenderMode::BGR);
        window_visu->set_keyboard_callback(
            [&window_visu](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
                if (action == Metavision::UIAction::RELEASE &&
                    (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                    window_visu->set_close_flag();
                }
            });
        window_legend =
            std::make_unique<Metavision::Window>("Flow Legend", 100, 100, Metavision::BaseWindow::RenderMode::BGR);
        window_legend->set_keyboard_callback(
            [&window_legend](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
                if (action == Metavision::UIAction::RELEASE &&
                    (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                    window_legend->set_close_flag();
                }
            });
    }

    std::unique_ptr<Metavision::OnDemandFrameGenerationAlgorithm> cd_framer;
    std::unique_ptr<Metavision::DenseFlowFrameGeneratorAlgorithm> flow_framer;
    cv::Mat legend_frame;
    if (enable_visu) {
        cd_framer   = std::make_unique<Metavision::OnDemandFrameGenerationAlgorithm>(width, height, 0,
                                                                                   Metavision::ColorPalette::Dark);
        flow_framer = std::make_unique<Metavision::DenseFlowFrameGeneratorAlgorithm>(
            width, height, args.visualization_flow_scale,
            Metavision::DenseFlowFrameGeneratorAlgorithm::AccumulationPolicy::Last);
        if (enable_display) {
            flow_framer->generate_legend_image(legend_frame);
        }
    }

    cv::Mat combined_frame(2 * height, width, CV_8UC3);
    cv::Mat cd_frame   = combined_frame(cv::Rect(0, 0, width, height));
    cv::Mat flow_frame = combined_frame(cv::Rect(0, height, width, height));
    Metavision::EventBufferReslicerAlgorithm reslicer(
        [&](Metavision::EventBufferReslicerAlgorithm::ConditionStatus s, Metavision::timestamp t, std::size_t n) {
            if (!enable_visu)
                return;
            cd_framer->generate(t, cd_frame, false);
            flow_framer->generate(flow_frame, false);
            if (enable_display) {
                window_visu->show(combined_frame);
                window_legend->show(legend_frame);
            }
            if (enable_video_writer) {
                video_writer->write(combined_frame);
            }
        });
    reslicer.set_slicing_condition(
        Metavision::EventBufferReslicerAlgorithm::Condition::make_n_us(args.processing_period));

    Metavision::timestamp ts_first = -1, ts_last = -1;
    std::vector<Metavision::EventCD> cd_buffer;
    std::vector<Metavision::EventOpticalFlow> flow_buffer;
    auto reslicer_process_events_cb = [&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        if (begin != end) {
            if (ts_first == -1)
                ts_first = begin->t;
            ts_last = std::prev(end)->t;
        }
        // Filter the events
        if (hardware_stc_enabled) {
            // When sensor's STC is available, we don't use software STC
            cd_buffer.insert(cd_buffer.end(), begin, end);
        } else {
            stc_algo->process_events(begin, end, std::back_inserter(cd_buffer));
        }
        // Run the flow algorithm
        switch (args.flow_type) {
        case DenseFlowType::PlaneFitting:
            plane_fitting_flow_algo->process_events(cd_buffer.cbegin(), cd_buffer.cend(),
                                                    std::back_inserter(flow_buffer));
            break;
        case DenseFlowType::TripletMatching:
            triplet_matching_flow_algo->process_events(cd_buffer.cbegin(), cd_buffer.cend(),
                                                       std::back_inserter(flow_buffer));
            break;
        default:
            throw std::runtime_error("Selected DenseFlowType is not implemented!");
        }
        // Process the visualization
        if (enable_visu) {
            cd_framer->process_events(cd_buffer.cbegin(), cd_buffer.cend());
            flow_framer->process_events(flow_buffer.cbegin(), flow_buffer.cend());
        }
        cd_buffer.clear();
        flow_buffer.clear();
    };

    // Vector of CD events and vector of OpticalFlow events to store the output of the two algorithms
    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        reslicer.process_events(begin, end, reslicer_process_events_cb);
    });

    const auto ts_system_start = std::chrono::high_resolution_clock::now();
    camera.start();
    while (camera.is_running()) {
        Metavision::EventLoop::poll_and_dispatch(20);
        if (enable_display && (window_visu->should_close() || window_legend->should_close())) {
            break;
        }
    }
    camera.stop();
    const auto ts_system_stop = std::chrono::high_resolution_clock::now();
    const auto elapsed        = std::chrono::duration_cast<std::chrono::milliseconds>(ts_system_stop - ts_system_start);

    MV_LOG_INFO() << "Ran in" << Metavision::Log::no_space << static_cast<float>(elapsed.count()) / 1000.f << "s";
    MV_LOG_INFO() << "Record duration " << Metavision::Log::no_space << static_cast<float>(ts_last - ts_first) / 1e6f
                  << "s";
    if (enable_video_writer) {
        video_writer->release();
        MV_LOG_INFO() << "Wrote video file to:" << args.out_avi_file_path;
    }

    return 0;
}
