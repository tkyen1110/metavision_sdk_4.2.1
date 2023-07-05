/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_LINE_PARTICLE_TRACK_DRAWING_HELPER_H
#define METAVISION_SDK_ANALYTICS_LINE_PARTICLE_TRACK_DRAWING_HELPER_H

#include <opencv2/core/mat.hpp>

#include "metavision/sdk/analytics/utils/line_particle_tracking_output.h"
#include "metavision/sdk/base/utils/object_pool.h"

namespace Metavision {

/// @brief Class that superimposes Particle Size Measurement results on an image filled with events
class LineParticleTrackDrawingHelper {
public:
    /// @brief Constructor
    /// @param width Sensor's width (in pixels)
    /// @param height Sensor's height (in pixels)
    /// @param persistence_time_us Time interval (in the events-clock) during which particle contours remain visible in
    /// the visualization. Since a track is sent only once, it will appear only on one frame if we don't keep results in
    /// memory. We need to know how long we want to display these detected tracks
    LineParticleTrackDrawingHelper(int width, int height, int persistence_time_us);

    /// @brief Stores and draws particle tracks
    /// @tparam InputTrackIt An iterator type over a track of type @ref LineParticleTrack
    /// @param ts Detection timestamp
    /// @param output_img Output image
    /// @param begin Begin iterator to the particle tracks to display
    /// @param end Past-end iterator to the particle tracks to display
    template<typename InputTrackIt>
    void draw(timestamp ts, cv::Mat &output_img, InputTrackIt begin, InputTrackIt end);

private:
    /// @brief Struct representing a particle track to draw
    struct TrackedParticle {
        void clear();

        cv::Point pt_begin;             ///< Center of the first particle detection
        cv::Point pt_end;               ///< Center of the last particle detection
        float particle_size;            ///< Estimated size of the particle
        std::vector<cv::Point> contour; ///< 2D Contour of the particle

        timestamp t; ///< Detection timestamp
    };

    /// @brief Shared pointer to an element of the object pool storing the particle tracks to draw
    using TrackedParticlerPtr = typename SharedObjectPool<TrackedParticle>::ptr_type;

    /// @brief Stores particle tracks
    /// @tparam InputTrackIt An iterator type over a track of type @ref LineParticleTrack
    /// @param ts Detection timestamp
    /// @param begin Begin iterator to the particle tracks to display
    /// @param end Past-end iterator to the particle tracks to display
    template<typename InputTrackIt>
    void add_new_tracks(timestamp ts, InputTrackIt begin, InputTrackIt end);

    void add_new_track(const LineParticleTrack &track);

    void remove_old_tracks(timestamp ts);

    void draw_internal(timestamp ts, cv::Mat &output_img);

    std::vector<TrackedParticlerPtr> tracks_to_draw_; ///< Tracked particles to draw
    SharedObjectPool<TrackedParticle> tracks_pool;    ///< Pool to avoid reallocating memory for new contours

    const int persistence_time_us_; ///< Time interval during which particle contours remain visible in the
                                    ///< visualization (Since this information is only sent once, we need to introduce
                                    ///< some sort of retinal persistence)

    const int width_;
    const int height_;

    // Colors
    const cv::Vec3b contour_color_   = cv::Vec3b(0, 255, 0);
    const cv::Vec3b traj_line_color_ = cv::Vec3b(0, 0, 255);
    const cv::Vec3b text_color_      = cv::Vec3b(255, 255, 255);
};

template<typename InputTrackIt>
void LineParticleTrackDrawingHelper::draw(timestamp ts, cv::Mat &output_img, InputTrackIt begin, InputTrackIt end) {
    assert(output_img.cols == width_);
    assert(output_img.rows == height_);

    // Store new particle contours
    add_new_tracks(ts, begin, end);

    // Remove tracks that have been displayed for too long
    remove_old_tracks(ts);

    // Draw active particle contours
    draw_internal(ts, output_img);
}

template<typename InputTrackIt>
void LineParticleTrackDrawingHelper::add_new_tracks(timestamp ts, InputTrackIt begin, InputTrackIt end) {
    // Store each new particle contour that isn't too old
    // and keep only the part visible on the camera image
    const timestamp ts_min = ts - persistence_time_us_;
    for (auto it = begin; it != end; ++it)
        if (it->t + 1 > ts_min)
            add_new_track(*it);
}

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_LINE_PARTICLE_TRACK_DRAWING_HELPER_H
