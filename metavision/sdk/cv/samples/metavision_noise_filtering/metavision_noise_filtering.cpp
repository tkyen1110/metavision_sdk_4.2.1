/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// This code sample demonstrates how to use Metavision SDK CV module to test different noise filtering strategies.
// In addition, it shows how to capture the keys pressed in a display window so as to modify the behavior of the sample
// while it is running.

#include <functional>
#include <boost/program_options.hpp>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/cv/algorithms/activity_noise_filter_algorithm.h>
#include <metavision/sdk/cv/algorithms/trail_filter_algorithm.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/utils/window.h>

// function that applies a filter on events if it's not a nullptr and boolean parameter apply is true
template<class AlgoPtr, class InputIt, class FilteredIt>
inline void apply_filter(const AlgoPtr &algo, InputIt &begin, InputIt &end, std::vector<FilteredIt> &output_buffer,
                         bool apply) {
    if (algo && apply) {
        assert(begin != nullptr && end != nullptr);
        output_buffer.clear();
        algo->process_events(begin, end, std::back_inserter(output_buffer));
        begin = nullptr;
        end   = nullptr;
    }
};

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string in_file_path;
    std::string active_filter;
    bool no_display              = false;
    bool realtime_playback_speed = true;
    bool enable_activity         = false;
    bool enable_trail            = false;
    bool enable_stc              = false;
    std::unique_ptr<Metavision::Window> window;

    const std::string short_program_desc(
        "Code sample showing how to create a simple application testing different noise filtering strategies.\n");

    const std::string long_program_desc(short_program_desc +
                                        "Available keyboard options:\n"
                                        "  - a - filter events using the activity noise filter algorithm\n"
                                        "  - t - filter events using the trail filter algorithm\n"
                                        "  - s - filter events using the spatio temporal contrast algorithm\n"
                                        "  - e - show all events\n"
                                        "  - q - quit the application\n"
                                        "  -ESC- quit the application\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-file,i", po::value<std::string>(&in_file_path), "Path to input file. If not specified, the camera live stream is used.")
        ("activate-filter,f", po::value<std::string>(&active_filter), "Filter to activate by default :[activity/trail/stc]")
        ("no-display,d", po::bool_switch(&no_display)->default_value(false), "Disable output display window")
        ("realtime-playback-speed", po::value<bool>(&realtime_playback_speed)->default_value(true), "Replay events at speed of recording if true, otherwise as fast as possible");
    // clang-format on

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << short_program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return 1;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << short_program_desc;
        MV_LOG_INFO() << options_desc;
        return 0;
    }

    MV_LOG_INFO() << long_program_desc;

    if (active_filter == "activity") {
        enable_activity = true;
    } else if (active_filter == "trail") {
        enable_trail = true;
    } else if (active_filter == "stc") {
        enable_stc = true;
    }

    // construct a camera from a recording or a live stream
    Metavision::Camera cam;
    if (!in_file_path.empty()) {
        cam = Metavision::Camera::from_file(in_file_path,
                                            Metavision::FileConfigHints().real_time_playback(realtime_playback_speed));
    } else {
        cam = Metavision::Camera::from_first_available();
    }

    try {
        // Set ERC to 20Mev/s
        cam.erc_module().set_cd_event_rate(20000000);
        cam.erc_module().enable(true);
    } catch (Metavision::CameraException &e) {}

    const unsigned short width  = cam.geometry().width();
    const unsigned short height = cam.geometry().height();

    auto activity_filter = std::make_unique<Metavision::ActivityNoiseFilterAlgorithm<>>(width, height, 20000);
    auto trail_filter    = std::make_unique<Metavision::TrailFilterAlgorithm>(width, height, 100000);
    auto stc_filter      = std::make_unique<Metavision::SpatioTemporalContrastAlgorithm>(width, height, 10000, true);

    Metavision::PeriodicFrameGenerationAlgorithm frame_gen(width, height, 30000);
    std::vector<Metavision::EventCD> output;

    // add callback that will pass the events to the filter (if enabled) and then the frame generator
    cam.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        apply_filter(activity_filter, begin, end, output, enable_activity);
        apply_filter(trail_filter, begin, end, output, enable_trail);
        apply_filter(stc_filter, begin, end, output, enable_stc);

        if (!enable_activity && !enable_trail && !enable_stc) {
            frame_gen.process_events(begin, end);
        } else {
            frame_gen.process_events(output.cbegin(), output.cend());
        }
    });

    // set window if not disabled
    if (!no_display) {
        window =
            std::make_unique<Metavision::Window>("CD events", width, height, Metavision::BaseWindow::RenderMode::BGR);
        window->set_keyboard_callback(
            [&](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
                if (action == Metavision::UIAction::RELEASE) {
                    switch (key) {
                    case Metavision::UIKeyEvent::KEY_A:
                        // enable the activity filter
                        MV_LOG_INFO() << "Set activity filter";
                        enable_activity = true;
                        enable_trail    = false;
                        enable_stc      = false;
                        break;
                    case Metavision::UIKeyEvent::KEY_T:
                        // enable the trail filter
                        MV_LOG_INFO() << "Set trail filter";
                        enable_activity = false;
                        enable_trail    = true;
                        enable_stc      = false;
                        break;
                    case Metavision::UIKeyEvent::KEY_S:
                        // enable the spatio temporal contrast filter
                        MV_LOG_INFO() << "Set spatio temporal contrast filter";
                        enable_activity = false;
                        enable_trail    = false;
                        enable_stc      = true;
                        break;
                    case Metavision::UIKeyEvent::KEY_E:
                        // show all events (no filtering enabled)
                        MV_LOG_INFO() << "Noise filtering disabled";
                        enable_activity = false;
                        enable_trail    = false;
                        enable_stc      = false;
                        break;
                    case Metavision::UIKeyEvent::KEY_Q:
                    case Metavision::UIKeyEvent::KEY_ESCAPE:
                        // quit the application
                        MV_LOG_INFO() << "Quit application";
                        window->set_close_flag();
                        break;
                    }
                }
            });
    }

    frame_gen.set_output_callback([&](Metavision::timestamp, cv::Mat &frame) {
        if (window) {
            window->show(frame);
        }
    });

    cam.start();

    while (cam.is_running()) {
        static constexpr std::int64_t kSleepPeriodMs = 20;
        Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
        if (window && window->should_close()) {
            break;
        }
    }
    cam.stop();

    MV_LOG_INFO() << "\rFinished processing input events ";

    return 0;
}
