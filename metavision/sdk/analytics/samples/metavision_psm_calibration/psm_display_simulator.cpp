/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/utils/colors.h>

#include "psm_display_simulator.h"

namespace Metavision {

PsmDisplaySimulator::PsmDisplaySimulator(int width, int height, const PsmDisplaySimulatorConfig &config,
                                         const std::vector<int> &rows, std::vector<EventCD> &&events) :
    width_(width), height_(height), rows_(rows), persistence_contour_(config.persistence_contour), updated_(false) {
    int num_process_before_matching;

    // Get background color
    const RGBColor col = Metavision::get_color(ColorPalette::Dark, ColorType::Background);
    bg_color_          = cv::Vec3b(static_cast<uchar>(col.b * 255 + 0.5), static_cast<uchar>(col.g * 255 + 0.5),
                          static_cast<uchar>(col.r * 255 + 0.5));

    // Initialize CSV config file
    csv_config_loader_ = std::make_unique<PsmConfigCsvLoader>(
        width_, height_, config.output_dir, config.avg_speed_pix_ms, config.avg_size_pix, !config.is_going_up,
        rows.front(), rows.back(), static_cast<int>(rows.size()));

    csv_config_loader_->load_config(cluster_config_, particle_config_, num_process_before_matching);

    // Initialize TimeSliceCursor (From events and config)
    timeslice_cursor_ =
        std::make_unique<TimeSliceCursor>(std::move(events), cluster_config_.precision_time_us_,
                                          cluster_config_.bitsets_buffer_size_, num_process_before_matching);

    // Initialize drawing helpers
    counting_drawing_helper_      = std::make_unique<CountingDrawingHelper>(rows);
    line_particle_drawing_helper_ = std::make_unique<LineParticleTrackDrawingHelper>(
        width_, height_, persistence_contour_ * cluster_config_.precision_time_us_);
    line_cluster_drawing_helper_ = std::make_unique<LineClusterDrawingHelper>();

    // Initialize the last callback results
    reset_psm_results();

    // Initialize the display
    back_img_.create(height_, width_, CV_8UC3);
    back_img_.setTo(bg_color_);

    // Run algo for the first time
    run(RunType::RERUN);
}

void PsmDisplaySimulator::run_at_given_ts(timestamp ts) {
    timeslice_cursor_->go_to(ts);
    run(RunType::RERUN);
}

void PsmDisplaySimulator::run(RunType run_type) {
    if (run_type == RunType::FORWARD)
        timeslice_cursor_->advance(true);
    else {
        switch (run_type) {
        case RunType::BACKWARD:
            timeslice_cursor_->advance(false);
            break;
        case RunType::BEGIN:
            timeslice_cursor_->go_to_begin();
            break;
        case RunType::END:
            timeslice_cursor_->go_to_end();
            break;
        default: // In case of RERUN, don't change the timeslice
            break;
        }

        // Reload config
        csv_config_loader_->load_config(cluster_config_, particle_config_, num_process_before_matching);
        timeslice_cursor_->update_parameters(cluster_config_.precision_time_us_, cluster_config_.bitsets_buffer_size_,
                                             num_process_before_matching);

        // Reset PSM algorithm
        psm_algo_ = std::make_unique<PsmAlgorithm>(width_, height_, rows_, cluster_config_, particle_config_,
                                                   num_process_before_matching);

        // Reset drawing helper
        line_particle_drawing_helper_ = std::make_unique<LineParticleTrackDrawingHelper>(
            width_, height_, persistence_contour_ * cluster_config_.precision_time_us_);

        reset_psm_results();

        psm_algo_->set_output_callback([this](const timestamp &ts, LineParticleTrackingOutput &tracks,
                                              PsmAlgorithm::LineClustersOutput &line_clusters) {
            // Take last timestamp
            last_ts_ = ts;

            // Take last line clusters
            last_line_clusters_ = line_clusters;

            // Accumulate tracks
            last_tracks_.global_counter = tracks.global_counter;
            tracks.buffer.move_and_insert_to(last_tracks_.buffer);
        });
    }

    // Regenerate the image to display with either new interval to process or new algorithm parameters
    process_events_and_update_image();
}

void PsmDisplaySimulator::process_events_and_update_image() {
    assert(psm_algo_);
    psm_algo_->process_events(timeslice_cursor_->ts(), timeslice_cursor_->cbegin(), timeslice_cursor_->cend());

    // Regenerate the image
    back_img_.create(height_, width_, CV_8UC3);
    back_img_.setTo(bg_color_);
    updated_ = true;

    if (last_ts_ <= 0) {
        reset_psm_results();
        last_ts_ = timeslice_cursor_->ts();
    }

    counting_drawing_helper_->draw(last_ts_, last_tracks_.global_counter, back_img_);
    line_cluster_drawing_helper_->draw(back_img_, last_line_clusters_.cbegin(), last_line_clusters_.cend());
    line_particle_drawing_helper_->draw(last_ts_, back_img_, last_tracks_.buffer.cbegin(), last_tracks_.buffer.cend());
    last_tracks_.clear(); // Clear tracks since they've already been added to the display
}

void PsmDisplaySimulator::swap_back_image_if_updated(cv::Mat &img) {
    if (updated_) {
        assert(!back_img_.empty());
        updated_ = false;
        cv::swap(back_img_, img);
    }
}

timestamp PsmDisplaySimulator::get_crt_ts() {
    return timeslice_cursor_->ts();
}

void PsmDisplaySimulator::reset_config() {
    csv_config_loader_->reset_config();
    run(RunType::RERUN);
}

bool PsmDisplaySimulator::is_at_the_end() const {
    return timeslice_cursor_->is_at_the_end();
}

void PsmDisplaySimulator::reset_psm_results() {
    last_ts_ = -1;

    last_tracks_.clear();
    last_tracks_.global_counter = 0;

    last_line_clusters_.clear();
}

} // namespace Metavision
