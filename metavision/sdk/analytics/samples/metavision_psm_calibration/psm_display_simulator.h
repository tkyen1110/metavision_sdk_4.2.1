/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef PSM_CALIBRATOR_H
#define PSM_CALIBRATOR_H

#include <memory>
#include <opencv2/core/mat.hpp>
#include <metavision/sdk/analytics/algorithms/psm_algorithm.h>
#include <metavision/sdk/analytics/utils/counting_drawing_helper.h>
#include <metavision/sdk/analytics/utils/line_particle_track_drawing_helper.h>
#include <metavision/sdk/analytics/utils/line_cluster_drawing_helper.h>
#include <metavision/sdk/base/events/event_cd.h>

#include "psm_config_csv_loader.h"
#include "time_slice_cursor.h"

namespace Metavision {

/// @brief Struct storing parameters to instantiate the PsmDisplaySimulator class
struct PsmDisplaySimulatorConfig {
    bool is_going_up;
    int persistence_contour;
    int avg_speed_pix_ms;
    int avg_size_pix;
    std::string output_dir;
};

/// @brief Class that allows visualizing on the fly the effects of a modification of the PSM algorithm's parameters
///
/// Once the class has been fed with events, the user can move the current timestamp and then retrieve the
/// image displaying the PSM results at this given timestamp. The possible timeslice updates are:
///   - move forward one step of time
///   - go back a step of time
///   - go to the beginning of the sequence
///   - go to the end of the sequence
///   - jump to a specific timestamp
///
/// As long as the timeslice is moving forward in time, we just need to process the missing events. However, if we go
/// back in time, we need to reprocess the entire sequence. Theoretically, we could modify the algorithm so that it's
/// able to forget the previous timestep, but it doesn't make sense to add complexity just for debugging reasons. So we
/// stick to the current public API of the algorithm and do our best with it. By the way, the algorithm is really fast
/// and we can easily afford to reprocess the sequence. We could only process the last n ms, but the total count of
/// detected particles since the beginning wouldn't be the right one, it would only represent a few detections.
///
/// The class suggests optimal PsmAlgorithm parameters and dumps them into a CSV, so that the user is free to
/// tune them and reload the parameters as many times as needed. Parameters are re-loaded from CSV whenever we reprocess
/// entirely the sequence, i.e. all the time except in mode FORWARD
class PsmDisplaySimulator {
public:
    /// @brief Type specifying how we want to generate the next image
    enum RunType {
        FORWARD,  ///< Move forward one step of time + Run algorithm on missing timestep
        BACKWARD, ///< Go back a step of time + RERUN
        BEGIN,    ///< Go to the beginning of the sequence + RERUN
        END,      ///< Go to the end of the sequence, reload config + RERUN
        RERUN     ///< Reload config, rerun algorithm
    };

    /// @brief Constructor
    /// @param width Sensor's width (in pixels)
    /// @param height Sensor's height (in pixels)
    /// @param config Configuration parameters
    /// @param rows Rows on which to instantiate line cluster trackers
    /// @param events Buffer of events, that the class is going to take ownership of
    PsmDisplaySimulator(int width, int height, const PsmDisplaySimulatorConfig &config, const std::vector<int> &rows,
                        std::vector<EventCD> &&events);

    /// @brief Updates timeslice and config to run the algorithm and update the result image
    /// @param run_type Run type
    void run(RunType run_type);

    /// @brief Moves timeslice to timestamp @p ts and reloads config to run the algorithm and update the result image
    /// @param ts Timestamp to reach
    void run_at_given_ts(timestamp ts);

    /// @brief Gets the result image by swap, provided that the image has been updated
    /// @param img Image to swap
    void swap_back_image_if_updated(cv::Mat &img);

    /// @brief Gets upperbound of the current timeslice, i.e. the one used to generate the result image
    /// @return Timestamp
    timestamp get_crt_ts();

    /// @brief Overwrites current CSV with the initial suggested PsmAlgorithm configuration
    void reset_config();

    /// @brief Checks if the timeslice is at the end of the sequence
    /// @return true if it's at the end
    bool is_at_the_end() const;

private:
    /// @brief Runs the algorithm on the timeslice and generates the new result image
    void process_events_and_update_image();

    /// @brief Reset PSM results
    void reset_psm_results();

    // Utils
    std::unique_ptr<PsmConfigCsvLoader> csv_config_loader_;
    std::unique_ptr<TimeSliceCursor> timeslice_cursor_;

    // PSM algorithm and drawing helpers
    std::unique_ptr<PsmAlgorithm> psm_algo_;                                       ///< Psm algorithm
    std::unique_ptr<CountingDrawingHelper> counting_drawing_helper_;               ///< Counting drawing helper
    std::unique_ptr<LineParticleTrackDrawingHelper> line_particle_drawing_helper_; ///< Psm Frame generator
    std::unique_ptr<LineClusterDrawingHelper> line_cluster_drawing_helper_;        ///< Psm Frame generator

    // Psm results
    timestamp last_ts_ = -1;
    LineParticleTrackingOutput last_tracks_;
    PsmAlgorithm::LineClustersOutput last_line_clusters_;

    // Test parameters
    LineClusterTrackingConfig cluster_config_;
    LineParticleTrackingConfig particle_config_;
    int num_process_before_matching;

    // Back image
    cv::Mat back_img_;
    bool updated_;

    // Constants
    cv::Vec3b bg_color_;
    const int width_;
    const int height_;
    const std::vector<int> rows_;
    const int persistence_contour_; ///< Number of frames during which particle contours remain visible in the
                                    ///< display (Since this information is only sent once, we need to introduce
                                    ///< some sort of retinal persistence)
                                    // Rows
};

} // namespace Metavision

#endif // PSM_CALIBRATOR_H
