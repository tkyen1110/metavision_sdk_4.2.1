/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// This code sample demonstrates how to use the Metavision SDK CV3D module to detect and track 2D edgelets in a time
// surface.

#include <boost/program_options.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/algorithms/shared_cd_events_buffer_producer_algorithm.h>
#include <metavision/sdk/core/utils/colors.h>
#include <metavision/sdk/core/utils/mostrecent_timestamp_buffer.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/cv3d/events/event_edgelet_2d.h>
#include <metavision/sdk/cv3d/algorithms/edgelet_2d_detection_algorithm.h>
#include <metavision/sdk/cv3d/utils/edgelet_utils.h>
#include <metavision/sdk/cv3d/algorithms/edgelet_2d_tracking_algorithm.h>

static constexpr int GRID_CELL_SIZE = 16; // [pixels]

static const cv::Vec3b COLOR_BG  = cv::Vec3b(52, 37, 30);
static const cv::Vec3b COLOR_ON  = cv::Vec3b(236, 223, 216);
static const cv::Vec3b COLOR_OFF = cv::Vec3b(201, 126, 64);

namespace po         = boost::program_options;
using SharedCDBuffer = Metavision::SharedCdEventsBufferProducerAlgorithm::SharedEventsBuffer;

int main(int argc, char *argv[]) {
    std::string in_raw_file_path;

    const std::string short_program_desc("Code sample showing how to detect 2D edgelets in a time surface\n");
    const std::string long_program_desc(short_program_desc + "Available keyboard options:\n"
                                                             "  - q - quit the application\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-raw-file,i", po::value<std::string>(&in_raw_file_path), "Path to input RAW file. If not specified, the camera live stream is used.")
        ;
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

    // Construct a camera from a recording or a live stream
    Metavision::Camera camera;
    if (!in_raw_file_path.empty()) {
        camera = Metavision::Camera::from_file(in_raw_file_path);
    } else {
        camera = Metavision::Camera::from_first_available();
    }

    const auto width  = camera.geometry().width();
    const auto height = camera.geometry().height();

    // The time surface
    Metavision::MostRecentTimestampBuffer time_surface(height, width, 2);
    time_surface.set_to(0);

    // The 2D edgelet detection algorithm
    Metavision::Edgelet2dDetectionAlgorithm edgelet_2d_detection_algo;

    // The 2D edgelet tracking algorithm
    Metavision::Edgelet2dTrackingAlgorithm edgelet_2d_tracking_algo;

    // We use a mask to avoid detecting edgelets at every event's location.
    const int grid_width  = std::ceil(width / static_cast<float>(GRID_CELL_SIZE));
    const int grid_height = std::ceil(height / static_cast<float>(GRID_CELL_SIZE));
    cv::Mat detection_mask(grid_height, grid_width, CV_8UC1);

    std::vector<Metavision::EventCD> events_tmp;               // Temporary buffer used to detect new edgelets
    std::vector<Metavision::EventEdgelet2d> edgelets_to_track; // Edgelets that have been tracked so far
    std::vector<Metavision::EventEdgelet2d> tmp_edgelets;      // Temporary buffer used for both tracking and detection
    std::vector<bool> tracking_statuses; // Indicates if the corresponding edgelet has been tracked or not

    // Image used for visualization
    cv::Mat time_surface_8uc3(height, width, CV_8UC3);

    char key                = 0;
    const auto cd_buffer_cb = [&](Metavision::timestamp ts, const SharedCDBuffer &b) {
        const auto cd_begin = b->cbegin();
        const auto cd_end   = b->cend();

        // Update the time surface
        for (auto it = cd_begin; it != cd_end; ++it)
            time_surface.at(it->y, it->x, it->p) = it->t;

        // Update the visualization image
        time_surface_8uc3.setTo(COLOR_BG);
        for (auto it = cd_begin; it != cd_end; ++it)
            time_surface_8uc3.at<cv::Vec3b>(it->y, it->x) = (it->p == 0) ? COLOR_OFF : COLOR_ON;

        // Track the edgelets
        const auto n_edgelets = edgelets_to_track.size();
        tracking_statuses.resize(n_edgelets, false);
        tmp_edgelets.resize(n_edgelets); // buffer of new tracked edgelets
        auto tracked_edglet_end =
            edgelet_2d_tracking_algo.process(time_surface, ts, edgelets_to_track.cbegin(), edgelets_to_track.cend(),
                                             tmp_edgelets.begin(), tracking_statuses.begin());

        const auto n_tracked = std::distance(tmp_edgelets.begin(), tracked_edglet_end);
        tmp_edgelets.resize(n_tracked);

        // Display in red edgelets that have not been tracked
        for (size_t i = 0; i < n_edgelets; ++i) {
            if (!tracking_statuses[i]) {
                const auto &e = edgelets_to_track[i];
                const cv::Point2f dir =
                    Metavision::edgelet_direction_from_normal<cv::Matx21f, cv::Point2f>(e.unit_norm2_);
                const cv::Point2f start = cv::Point2f(e.ctr2_(0), e.ctr2_(1)) + 3 * dir;
                const cv::Point2f end   = cv::Point2f(e.ctr2_(0), e.ctr2_(1)) - 3 * dir;

                cv::line(time_surface_8uc3, start, end, CV_RGB(255, 0, 0), 2);
            }
        }

        // Display in green edgelets that have been tracked
        for (const auto &e : tmp_edgelets) {
            const cv::Point2f dir = Metavision::edgelet_direction_from_normal<cv::Matx21f, cv::Point2f>(e.unit_norm2_);
            const cv::Point2f start = cv::Point2f(e.ctr2_(0), e.ctr2_(1)) + 3 * dir;
            const cv::Point2f end   = cv::Point2f(e.ctr2_(0), e.ctr2_(1)) - 3 * dir;

            cv::line(time_surface_8uc3, start, end, CV_RGB(0, 255, 0), 2);
        }

        std::swap(edgelets_to_track, tmp_edgelets);

        // Don't detect new edgelets in cells where some are already tracked
        detection_mask.setTo(0);
        for (const auto &e : edgelets_to_track) {
            const int cellx = e.ctr2_(0) / GRID_CELL_SIZE;
            const int celly = e.ctr2_(1) / GRID_CELL_SIZE;

            detection_mask.at<uchar>(celly, cellx) = 255;
        }

        events_tmp.clear(); // buffer of events whose location will be used to detect new edgelets
        // We start from the end to only keep the last events
        for (auto it = b->crbegin(); it != b->crend(); ++it) {
            const int cellx = it->x / GRID_CELL_SIZE;
            const int celly = it->y / GRID_CELL_SIZE;

            if (detection_mask.at<uchar>(celly, cellx) == 0) {
                detection_mask.at<uchar>(celly, cellx) = 255;

                events_tmp.emplace_back(*it);
            }
        }

        tmp_edgelets.resize(events_tmp.size()); // buffer of new detected edgelets

        // Try to detect 2D edgelets at every event's location in the grid
        auto edgelet_end = edgelet_2d_detection_algo.process(time_surface, events_tmp.cbegin(), events_tmp.cend(),
                                                             tmp_edgelets.begin());

        tmp_edgelets.resize(std::distance(tmp_edgelets.begin(), edgelet_end));

        // Display in blue the new detected edgelets
        for (const auto &e : tmp_edgelets) {
            const cv::Point2f dir = Metavision::edgelet_direction_from_normal<cv::Matx21f, cv::Point2f>(e.unit_norm2_);
            const cv::Point2f start = cv::Point2f(e.ctr2_(0), e.ctr2_(1)) + 3 * dir;
            const cv::Point2f end   = cv::Point2f(e.ctr2_(0), e.ctr2_(1)) - 3 * dir;

            cv::line(time_surface_8uc3, start, end, CV_RGB(0, 0, 255), 2);
        }

        // Add the new detected edgelets to the tracked edgelets buffer
        edgelets_to_track.insert(edgelets_to_track.cend(), tmp_edgelets.cbegin(), tmp_edgelets.cend());

        cv::imshow("time surface", time_surface_8uc3);
        key = cv::waitKey(1);
    };

    // We will use a STC to filter events
    Metavision::SpatioTemporalContrastAlgorithm stc_filter(width, height, 5000);
    std::vector<Metavision::EventCD> stc_filtered_events;

    // We will produce an events time slice every 5000 events
    Metavision::SharedEventsBufferProducerParameters cd_producer_params;
    cd_producer_params.buffers_time_slice_us_ = 0;
    cd_producer_params.buffers_events_count_  = 5000;
    Metavision::SharedCdEventsBufferProducerAlgorithm cd_producer(cd_producer_params, cd_buffer_cb);

    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        stc_filtered_events.resize(std::distance(begin, end));

        auto stc_end = stc_filter.process_events(begin, end, stc_filtered_events.begin());
        stc_filtered_events.resize(std::distance(stc_filtered_events.begin(), stc_end));

        cd_producer.process_events(stc_filtered_events.cbegin(), stc_filtered_events.cend());
    });

    camera.start();

    while (camera.is_running() && key != 'q') {}

    camera.stop();

    return 0;
}