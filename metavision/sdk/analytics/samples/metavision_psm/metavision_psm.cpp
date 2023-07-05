/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <iostream>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <boost/program_options.hpp>
#include <opencv2/core/mat.hpp>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/analytics/algorithms/psm_algorithm.h>
#include <metavision/sdk/analytics/utils/counting_drawing_helper.h>
#include <metavision/sdk/analytics/utils/line_particle_track_drawing_helper.h>
#include <metavision/sdk/analytics/utils/line_cluster_drawing_helper.h>
#include <metavision/sdk/analytics/utils/sliding_histogram.h>
#include <metavision/sdk/analytics/utils/histogram_drawing_helper.h>
#include <metavision/sdk/analytics/utils/histogram_utils.h>
#include <metavision/sdk/base/events/event2d.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/cv/algorithms/activity_noise_filter_algorithm.h>
#include <metavision/sdk/cv/algorithms/transpose_events_algorithm.h>
#include <metavision/sdk/core/algorithms/polarity_filter_algorithm.h>
#include <metavision/sdk/core/algorithms/on_demand_frame_generation_algorithm.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

#include "simple_video_writer.h"

namespace bpo = boost::program_options;

using timestamp = Metavision::timestamp;

using LineParticleTrackingOutput = Metavision::LineParticleTrackingOutput;
using LineClustersOutput         = Metavision::PsmAlgorithm::LineClustersOutput;
using OutputCb = std::function<void(const timestamp, LineParticleTrackingOutput &, LineClustersOutput &)>;

class Pipeline {
public:
    Pipeline() = default;

    ~Pipeline() = default;

    /// @brief Utility function to parse command line attributes
    bool parse_command_line(int argc, char *argv[]);

    /// @brief Initialize the camera
    bool initialize_camera();

    /// @brief Initialize the trackers
    bool initialize_trackers();

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
    void increment_callback(const timestamp ts, int global_counter);

    /// @brief Function that is called every time no object is counted during a certain time. This time could be chosen
    /// with the parameter inactivity time.
    void inactivity_callback(const timestamp ts, const timestamp last_count_ts);

    /// @brief Processing of the output of the Particle Size Measurement
    void psm_callback(const timestamp &ts, LineParticleTrackingOutput &tracks, LineClustersOutput &line_clusters);

    /// @brief Processing of the events coming from the camera
    void camera_callback(const Metavision::EventCD *begin, const Metavision::EventCD *end);

    /// @brief Applies a series of drawing helpers to draw the events, the lines and the particles
    void draw_events_and_line_particles(timestamp ts, cv::Mat &events_img, const LineParticleTrackingOutput &tracks,
                                        const LineClustersOutput &line_clusters);

    /// @brief Callback called by the @ref Metavision::Window when a key is pressed
    void keyboard_callback(Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods);

    // Algorithm - Detection
    int cluster_ths_;      ///< Minimum width (in pixels) below which clusters of events are considered as noise
    int num_clusters_ths_; ///< Minimum number of cluster measurements below which a particle is considered as noise
    int min_inter_clusters_distance_; ///< Once small clusters have been removed, merge clusters that are closer than
                                      ///< this distance. This helps dealing with dead pixels that could cut particles
                                      ///< in half. If set to 0, do nothing
    float learning_rate_;  ///< Ratio in the weighted mean between the current x position and the observation. This is
                           ///< used only when the particle is shrinking, because the front of the particle is always
                           ///< sharp while the trail might be noisy. 0.0 is conservative and does not take the
                           ///< observation into account, whereas 1.0 has no memory and overwrites the cluster estimate
                           ///< with the new observation. A value outside ]0,1] disables the weighted mean, and 1.0 is
                           ///< used instead
    float max_dx_allowed_; ///< Caps x variation at this value. A negative value disables the clamping. This is used
                           ///< only when the particle is shrinking, because the front of the particle is always sharp
                           ///< while the trail might be noisy.

    // Algorithm - Tracking
    int min_y_line_;         ///< Min y to place the first line
    int max_y_line_;         ///< Max y to place the last line
    int num_lines_;          ///< Number of lines for processing between min-y and max-y
    bool is_going_up_;       ///< Set to true for upward motions, and false for downward motions
    int dt_first_match_ths_; ///< Maximum allowed duration to match the 2nd particle of a track
    int max_angle_deg_;  ///< Angle with the vertical beyond which two particles on consecutive lines can't be matched
    float matching_ths_; ///< Minimum similarity score in [0,1] needed to match two particles

    // Notifications
    int last_count_callback_ = 0;
    int inactivity_time_;       ///< Time of inactivity in us (no counter increment) to be notified
    int notification_sampling_; ///< Minimal number of counted objects between each notification
    timestamp last_ts_          = -1;
    timestamp last_ts_callback_ = -1;

    // Histogram
    std::vector<float> hist_bins_boundaries_;
    std::vector<float> hist_bins_centers_;
    std::vector<unsigned int> hist_counts_;
    int hist_min_;
    int hist_max_;
    int hist_width_bin_;
    bool disable_histogram_ = false;

    // Filtering
    std::vector<Metavision::EventCD> buffer_filters_;
    short polarity_;              ///< Process only events of this polarity
    timestamp activity_time_ths_; ///< Length of the time window for activity filtering (in us)

    // Camera parameters
    std::string filename_    = "";
    std::string biases_file_ = "";
    int roi_line_half_width_;

    // Display parameters
    bool no_display_          = false; ///< If display data on the screen or not
    bool as_fast_as_possible_ = false; ///< If display as fast as possible
    std::string output_video_ = "";    ///< Filename to save the resulted video
    cv::Mat back_img_;                 ///< Current image
    cv::Rect events_img_roi_;
    cv::Rect histogram_img_roi_;

    // Visualization
    bool transpose_axis_;     ///< Set to true to rotate the camera 90 degrees clockwise in case of particles moving
                              ///< horizontally in FOV
    int persistence_contour_; ///< Number of frames during which particle contours remain visible in the
                              ///< display (Since this information is only sent once, we need to introduce
                              ///< some sort of retinal persistence)

    // Conditional variables to notify the end of the processing
    std::condition_variable process_cond_;
    std::mutex process_mutex_;
    volatile bool is_processing_ = true;

    // Time
    timestamp precision_time_us_, accumulation_time_us_; ///< Parameters of the frame generation algo
    timestamp process_from_ = 0;  ///< Start time to process events and write the output video (in us)
    timestamp process_to_   = -1; ///< End time to process events and write the output video (in us)

    std::unique_ptr<Metavision::Camera> camera_;                                            ///< Camera
    std::unique_ptr<Metavision::PsmAlgorithm> psm_tracker_;                                 ///< Psm tracker
    std::unique_ptr<Metavision::CountingDrawingHelper> counting_drawing_helper_;            ///< Counting drawing helper
    std::unique_ptr<Metavision::OnDemandFrameGenerationAlgorithm> events_frame_generation_; ///< Events Frame generator
    std::unique_ptr<Metavision::LineParticleTrackDrawingHelper> line_particle_drawing_helper_; ///< Psm Frame generator
    std::unique_ptr<Metavision::LineClusterDrawingHelper> line_cluster_drawing_helper_;        ///< Psm Frame generator
    std::unique_ptr<Metavision::HistogramDrawingHelper> histogram_drawing_helper_;             ///< Psm Frame generator
    std::unique_ptr<SimpleVideoWriter> video_writer_;                                          ///< Video writer
    std::unique_ptr<Metavision::Window> window_;                                               ///< Display window
    std::unique_ptr<Metavision::PolarityFilterAlgorithm> polarity_filter_;                     ///< Filter by polarity
    std::unique_ptr<Metavision::TransposeEventsAlgorithm> transpose_events_filter_;     ///< Transpose X/Y on events
    std::unique_ptr<Metavision::ActivityNoiseFilterAlgorithm<>> activity_noise_filter_; ///< Filter noisy events
};

bool Pipeline::initialize_camera() {
    // If the filename is set, then read from the file
    if (filename_ != "") {
        try {
            camera_ = std::make_unique<Metavision::Camera>(Metavision::Camera::from_file(
                filename_, Metavision::FileConfigHints().real_time_playback(!as_fast_as_possible_)));
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

            // Set lines of interest
            if (roi_line_half_width_ >= 0) {
                std::vector<int> expanded_detection_indices; // Vector to fill with line counters ordinates +
                expanded_detection_indices.reserve(num_lines_ * (1 + 2 * roi_line_half_width_));

                const int y_line_step = (max_y_line_ - min_y_line_) / (num_lines_ - 1);
                for (int i = 0; i < num_lines_; ++i) {
                    const int line_ordinate = min_y_line_ + y_line_step * i;
                    for (int shift = -roi_line_half_width_; shift < roi_line_half_width_ + 1; shift++)
                        expanded_detection_indices.push_back(line_ordinate + shift);
                }

                std::vector<bool> cols_to_enable, rows_to_enable;
                if (transpose_axis_) {
                    rows_to_enable.resize(camera_->geometry().width(), true); // Use all rows
                    cols_to_enable.resize(camera_->geometry().height(), false);
                    for (int id : expanded_detection_indices)
                        cols_to_enable[id] = true;

                } else {
                    cols_to_enable.resize(camera_->geometry().width(), true); // Use all columns
                    rows_to_enable.resize(camera_->geometry().height(), false);
                    for (int id : expanded_detection_indices)
                        rows_to_enable[id] = true;
                }

                camera_->roi().set(cols_to_enable, rows_to_enable);

                std::stringstream ss;
                for (const auto &y : expanded_detection_indices)
                    ss << y << ", ";
                MV_LOG_INFO() << (transpose_axis_ ? "Columns" : "Rows") << " of interest are located at: [" << ss.str()
                              << "]";
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
        "Code sample for Particle Size Measurement algorithm on a stream of events from an event-based device or "
        "recorded data.\n");

    const std::string long_program_desc(
        short_program_desc +
        "By default, this samples uses only OFF events and assumes that the particles are moving vertically in FOV.\n"
        "In case of different configuration, the default parameters should be adjusted.\n"
        "Please note that the GUI is displayed only when reading RAW files "
        "as it would not make sense to generate frames at such high frequency\n\n"
        "Press 'q' or Escape key to leave the program.\n"
        "Press 'r' to reset the counter.\n"
        "Press 'p' to increase (+1) notification sampling (number of objects to be counted between each "
        "notification).\n"
        "Press 'm' to decrease (-1) notification sampling (number of objects to be counted between each "
        "notification).\n");

    bpo::options_description options_desc;
    bpo::options_description base_options("Base options");
    // clang-format off
    base_options.add_options()
        ("help,h", "Produce help message.")
        ("input-file,i",     bpo::value<std::string>(&filename_), "Path to input file. If not specified, the camera live stream is used.")
        ("biases,b",         bpo::value<std::string>(&biases_file_), "Path to a biases file. If not specified, the camera will be configured with the default biases.")
        ("process-from,s",   bpo::value<timestamp>(&process_from_), "Start time to process events (in us).")
        ("process-to,e",     bpo::value<timestamp>(&process_to_), "End time to process events (in us).")
        ("roi-line",         bpo::value<int>(&roi_line_half_width_)->default_value(-1), "If specified (greater than or equal to 0), it sets hardware lines of interest with a live camera; if not, "
                                                                                        "all the events are used. It corresponds to the half-width of each counting lines of interest to set, i.e. "
                                                                                        "the number of rows of pixels to add on each side of a line of interest to make the ROI thicker.")
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

    bpo::options_description detection_options("Detection options");
    // clang-format off
    detection_options.add_options()
        ("accumulation-time,a",     bpo::value<timestamp>(&accumulation_time_us_)->default_value(200), "Accumulation time in us (temporal length of the processed event-buffers).")
        ("precision,p",             bpo::value<timestamp>(&precision_time_us_)->default_value(30), "Precision time in us (time duration between two asynchronous processes).")
        ("cluster-ths",             bpo::value<int>(&cluster_ths_)->default_value(3), "Minimum width (in pixels) below which clusters of events along the line are considered as noise.")
        ("num-clusters-ths",        bpo::value<int>(&num_clusters_ths_)->default_value(4), "Minimum number of cluster measurements below which a particle is considered as noise.")
        ("min-inter-clusters-dist", bpo::value<int>(&min_inter_clusters_distance_)->default_value(1), "Once small clusters have been removed, merge clusters that are closer than this distance.")
        ("learning-rate",           bpo::value<float>(&learning_rate_)->default_value(0.8f), "Ratio in the weighted mean between the current x position and the observation."
                                                                                                  " This is used only when the particle is shrinking. 0.0 is conservative and does not take the observation"
                                                                                                  " into account, whereas 1.0 has no memory and overwrites the cluster estimate with the new observation.")
        ("clamping",                bpo::value<float>(&max_dx_allowed_)->default_value(5.f), "Caps x variation at this value. A negative value disables the clamping. This is used only when the particle is shrinking.")
        ;
    // clang-format on

    bpo::options_description tracking_options("Tracking options");
    // clang-format off
    tracking_options.add_options()
        ("min-y",               bpo::value<int>(&min_y_line_)->default_value(200), "Min y to place the first line cluster tracker.")
        ("max-y",               bpo::value<int>(&max_y_line_)->default_value(300), "Max y to place the last line cluster tracker.")
        ("num-lines,n",         bpo::value<int>(&num_lines_)->default_value(6), "Number of lines for processing between min-y and max-y.")
        ("objects-moving-up,u", bpo::bool_switch(&is_going_up_)->default_value(false), "Specify if the particles are going upwards.")
        ("first-match-dt",      bpo::value<int>(&dt_first_match_ths_)->default_value(100000), "Maximum allowed duration to match the 2nd particle of a track.")
        ("max-angle-deg",       bpo::value<int>(&max_angle_deg_)->default_value(45), "Angle with the vertical beyond which two particles on consecutive lines can't be matched.")
        ("matching-ths",        bpo::value<float>(&matching_ths_)->default_value(0.5f), "Minimum similarity score in [0,1] needed to match two particles.")
        ;
    // clang-format on

    bpo::options_description histogram_options("Histogram options");
    // clang-format off
    histogram_options.add_options()
        ("disable-hist", bpo::bool_switch(&disable_histogram_), "Disable the computation of the histogram of the particle sizes.")
        ("min-hist",     bpo::value<int>(&hist_min_)->default_value(10), "Lower bound of the histogram bins (minimum particle size).")
        ("max-hist",     bpo::value<int>(&hist_max_)->default_value(95), "Upper bound of the histogram bins (maximum particle size).")
        ("step-hist",    bpo::value<int>(&hist_width_bin_)->default_value(5), "Width of the bins of the histogram.")
        ;
    // clang-format on

    bpo::options_description outcome_options("Outcome options");
    // clang-format off
    outcome_options.add_options()
        ("no-display",            bpo::bool_switch(&no_display_), "Disable the GUI when reading a file (no effect with a live camera where GUI is already disabled).")
        ("out-video,o",           bpo::value<std::string>(&output_video_), "Path to an output AVI file to the resulting save slow motion video. A frame is generated after each process of the algorithm. The video "
                                                                           "will be written only for processed events. When the display is disabled, i.e. either with a live camera or when --no-display "
                                                                           "has been specified, frames are not generated, so the video can't be generated either.")
        ("persistence-contour",   bpo::value<int>(&persistence_contour_)->default_value(40), "Once a particle contour has been estimated, keep the drawing superimposed on the display for a given number of frames.")
        ("notification-sampling", bpo::value<int>(&notification_sampling_)->default_value(1), "Minimal number of counted objects between each notification.")
        ("inactivity-time",       bpo::value<int>(&inactivity_time_)->default_value(static_cast<int>(1e6)), "Time of inactivity in us (no counter increment) to be notified.")
        ;
    // clang-format on

    options_desc.add(base_options)
        .add(detection_options)
        .add(tracking_options)
        .add(filter_options)
        .add(histogram_options)
        .add(outcome_options);

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
    const bool is_offline = !filename_.empty();
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

    // Check polarity
    if (polarity_ != 1 && polarity_ != 0 && polarity_ != -1) {
        MV_LOG_ERROR() << "The polarity is not valid:" << polarity_;
        return false;
    }

    // Check positions of the line counters
    if (max_y_line_ <= min_y_line_) {
        MV_LOG_ERROR() << "The range of y-positions for the line counters is not valid :" << Metavision::Log::no_space
                       << "[" << min_y_line_ << ", " << max_y_line_ << "].";
        return false;
    }

    // Histogram
    if (hist_max_ <= hist_min_) {
        MV_LOG_ERROR() << "The histogram bins are not valid :"
                       << "[" << hist_min_ << ", " << hist_max_ << "].";
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

void Pipeline::increment_callback(const timestamp ts, int global_counter) {
    if (global_counter >= last_count_callback_ + notification_sampling_) {
        last_count_callback_ +=
            ((global_counter - last_count_callback_) / notification_sampling_) * notification_sampling_;
        MV_LOG_INFO() << "At" << ts << "the counter is" << global_counter;
    }
}

void Pipeline::inactivity_callback(const timestamp ts, const timestamp last_count_ts) {
    last_ts_callback_ = std::max(last_count_ts, last_ts_callback_);
    if (ts >= last_ts_callback_ + inactivity_time_) {
        last_ts_callback_ += ((ts - last_ts_callback_) / inactivity_time_) * inactivity_time_;
        MV_LOG_INFO() << "At" << ts << "inactivity period.";
    }
}

/// [PSM_CALLBACK_BEGIN]
void Pipeline::psm_callback(const timestamp &ts, LineParticleTrackingOutput &tracks,
                            LineClustersOutput &line_clusters) {
    if (counting_drawing_helper_) {
        back_img_.create(events_img_roi_.height, events_img_roi_.width + histogram_img_roi_.width, CV_8UC3);
        if (!histogram_drawing_helper_) {
            draw_events_and_line_particles(ts, back_img_, tracks, line_clusters);
        } else {
            for (auto it = tracks.buffer.cbegin(); it != tracks.buffer.cend(); it++) {
                size_t id_bin;
                if (Metavision::value_to_histogram_bin_id(hist_bins_boundaries_, it->particle_size, id_bin))
                    hist_counts_[id_bin]++;
            }

            if (!tracks.buffer.empty()) {
                int count      = 0;
                using SizeType = std::vector<unsigned int>::size_type;
                for (SizeType k = 0; k < hist_counts_.size(); k++) {
                    count += hist_counts_[k];
                }
                if (count != 0) {
                    MV_LOG_INFO() << "Histogram : " << count;
                    std::stringstream ss;
                    for (SizeType k = 0; k < hist_counts_.size(); k++) {
                        if (hist_counts_[k] != 0)
                            ss << hist_bins_centers_[k] << ":" << hist_counts_[k] << " ";
                    }
                    MV_LOG_INFO() << ss.str();
                }
            }
            back_img_.create(events_img_roi_.height, events_img_roi_.width + histogram_img_roi_.width, CV_8UC3);
            auto events_img = back_img_(events_img_roi_);
            draw_events_and_line_particles(ts, events_img, tracks, line_clusters);
            auto hist_img = back_img_(histogram_img_roi_);
            histogram_drawing_helper_->draw(hist_img, hist_counts_);
        }

        if (video_writer_)
            video_writer_->write_frame(ts, back_img_);

        if (window_) {
            window_->show(back_img_);
        }
    }

    current_time_callback(ts);
    increment_callback(ts, tracks.global_counter);
    inactivity_callback(ts, tracks.last_count_ts);
}
/// [PSM_CALLBACK_END]

void Pipeline::draw_events_and_line_particles(timestamp ts, cv::Mat &events_img,
                                              const LineParticleTrackingOutput &tracks,
                                              const LineClustersOutput &line_clusters) {
    events_frame_generation_->generate(ts, events_img, false);
    counting_drawing_helper_->draw(ts, tracks.global_counter, events_img);
    if (line_particle_drawing_helper_) {
        line_cluster_drawing_helper_->draw(events_img, line_clusters.cbegin(), line_clusters.cend());
        line_particle_drawing_helper_->draw(ts, events_img, tracks.buffer.cbegin(), tracks.buffer.cend());
    }
}

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
    apply_algorithm_if_enabled(psm_tracker_, begin, end, buffer_filters_, is_first);
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
            break;
        }
        case Metavision::UIKeyEvent::KEY_R:
            MV_LOG_INFO() << "Reset trackers";
            psm_tracker_->reset();
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

bool Pipeline::initialize_trackers() {
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
        MV_LOG_ERROR() << "Error : min-y should be positive.";
        MV_LOG_ERROR() << "        min-y passed : " << min_y_line_;
        return false;
    }
    if (max_y_line_ > sensor_height) {
        MV_LOG_ERROR() << "Error : max-y should be smaller than the sensor's height (" << sensor_height << ").";
        MV_LOG_ERROR() << "        max-y passed : " << max_y_line_;
        return false;
    }

    // Activity Noise Filter
    if (activity_time_ths_ != 0) {
        activity_noise_filter_ = std::make_unique<Metavision::ActivityNoiseFilterAlgorithm<>>(
            sensor_width, sensor_height, activity_time_ths_);
    }

    if (polarity_ >= 0) {
        polarity_filter_ = std::make_unique<Metavision::PolarityFilterAlgorithm>(polarity_);
    }

    /// [PSM_ALGORITHM_INSTANTIATION_BEGIN]
    // PsmAlgorithm
    std::vector<int> rows; // Vector to fill with line counters ordinates
    rows.reserve(num_lines_);
    const int y_line_step = (max_y_line_ - min_y_line_) / (num_lines_ - 1);
    for (int i = 0; i < num_lines_; ++i) {
        const int line_ordinate = min_y_line_ + y_line_step * i;
        rows.push_back(line_ordinate);
    }

    const int bitsets_buffer_size = static_cast<int>(accumulation_time_us_ / precision_time_us_);
    const int num_clusters_ths = 7; ///< Min nbr of cluster measurements below which a particle is considered as noise
    Metavision::LineClusterTrackingConfig detection_config(
        static_cast<unsigned int>(precision_time_us_), bitsets_buffer_size, cluster_ths_, num_clusters_ths_,
        min_inter_clusters_distance_, learning_rate_, max_dx_allowed_, 0);

    Metavision::LineParticleTrackingConfig tracking_config(!is_going_up_, dt_first_match_ths_,
                                                           std::tan(max_angle_deg_ * 3.14 / 180.0), matching_ths_);

    const int num_process_before_matching = 3; // Accumulate particle detections during n process
                                               // before actually matching them to existing trajectories

    psm_tracker_ = std::make_unique<Metavision::PsmAlgorithm>(sensor_width, sensor_height, rows, detection_config,
                                                              tracking_config, num_process_before_matching);
    /// [PSM_ALGORITHM_INSTANTIATION_END]

    std::stringstream ss;
    for (const auto &y : rows)
        ss << y << ", ";
    MV_LOG_INFO() << "Tracking lines are located at: [" << ss.str() << "]";

    if (!output_video_.empty() || !no_display_) {
        counting_drawing_helper_ = std::make_unique<Metavision::CountingDrawingHelper>(rows);

        line_particle_drawing_helper_ = std::make_unique<Metavision::LineParticleTrackDrawingHelper>(
            sensor_width, sensor_height, static_cast<int>(persistence_contour_ * precision_time_us_));
        line_cluster_drawing_helper_ = std::make_unique<Metavision::LineClusterDrawingHelper>();

        int total_width = sensor_width;
        if (!disable_histogram_) {
            Metavision::init_histogram_bins<float>(static_cast<float>(hist_min_), static_cast<float>(hist_max_),
                                                   static_cast<float>(hist_width_bin_), hist_bins_centers_,
                                                   hist_bins_boundaries_);
            hist_counts_.resize(hist_bins_centers_.size(), 0);

            histogram_drawing_helper_ =
                std::make_unique<Metavision::HistogramDrawingHelper>(sensor_height, hist_bins_centers_);

            const int hist_width = histogram_drawing_helper_->get_width();
            histogram_img_roi_   = cv::Rect(0, 0, hist_width, sensor_height);
            events_img_roi_      = cv::Rect(hist_width, 0, sensor_width, sensor_height);

            total_width += hist_width;
        } else {
            events_img_roi_    = cv::Rect(0, 0, sensor_width, sensor_height);
            histogram_img_roi_ = cv::Rect(0, 0, 0, 0);
        }

        events_frame_generation_.reset(new Metavision::OnDemandFrameGenerationAlgorithm(
            sensor_width, sensor_height, static_cast<std::uint32_t>(accumulation_time_us_)));

        if (!output_video_.empty()) {
            video_writer_ = std::make_unique<SimpleVideoWriter>(
                total_width, sensor_height, static_cast<int>(precision_time_us_), 30.f, output_video_);

            video_writer_->set_write_range(process_from_, process_to_);
        }

        if (!no_display_) {
            window_ = std::make_unique<Metavision::Window>(
                "Particle Size Measurement Sample. Press 'q' or Escape key to exit.", sensor_width, sensor_height,
                Metavision::BaseWindow::RenderMode::BGR);

            window_->set_keyboard_callback(std::bind(&Pipeline::keyboard_callback, this, std::placeholders::_1,
                                                     std::placeholders::_2, std::placeholders::_3,
                                                     std::placeholders::_4));
        }
    }

    psm_tracker_->set_output_callback(
        [this](const timestamp &ts, LineParticleTrackingOutput &tracks, LineClustersOutput &line_clusters) {
            psm_callback(ts, tracks, line_clusters);
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

    // Initialize trackers
    if (!pipeline.initialize_trackers())
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
