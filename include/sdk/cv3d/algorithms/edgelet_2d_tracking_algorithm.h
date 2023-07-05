/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV3D_EDGELET_2D_TRACKING_ALGORITHM_H
#define METAVISION_SDK_CV3D_EDGELET_2D_TRACKING_ALGORITHM_H

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/core/utils/mostrecent_timestamp_buffer.h"

namespace Metavision {

/// @brief Algorithm used to track 2D edgelets in a time surface
///
/// Points are sampled along the edgelet's direction (a.k.a support points) and matches are looked for on the two sides
/// of the edgelet along its normal. A match is found when a timestamp more recent than a target timestamp if found in
/// the time surface. A new line is then fitted from those matches and both a new direction and normal are computed.
///
/// @warning Because of the aperture problem, the tracking can drift very quickly. As a result, this algorithm should
/// only be used if there are methods that constrain the tracking or to track edgelets that are orthogonal to the
/// camera's motion.
class Edgelet2dTrackingAlgorithm {
public:
    /// @brief Parameters used by the tracking algorithm
    struct Parameters {
        Parameters(){}; // We have to define a default constructor like this because of a bug in GCC & clang

        /// Radius in which matches are looked for each edgelet's side
        unsigned int search_radius_ = 3;

        /// Number of points sampled along the edgelet's direction
        unsigned int n_support_points_ = 3;

        /// Distance in pixels between the sampled points
        float support_points_distance_ = 2;

        /// Time tolerance used in the tracking
        timestamp threshold_ = 3000;

        /// Distance to the median position of the support points' matches above which a match is considered as outlier
        unsigned int median_outlier_threshold_ = 1;
    };

    /// @brief Constructor
    /// @param params Parameters used by the tracking algorithm
    Edgelet2dTrackingAlgorithm(const Parameters &params = Parameters());

    /// @brief Destructor
    ~Edgelet2dTrackingAlgorithm() = default;

    /// @brief Tracks the input 2D edgelets in the input time surface
    ///
    /// For each input 2D edgelet a status is updated to indicate whether the corresponding edgelet has been tracked or
    /// not. When an edgelet is successfully tracked, an updated 2D edgelet (i.e. corresponding to the matched edgelet
    /// in the time surface) is outputted. This means that a user needs to either, pass a std::back_insert_iterator to
    /// these two buffers, or, pre-allocate them with the same size as the input 2D edgelets buffer's one.
    ///
    /// @tparam Edgelet2dIt Iterator type of the input 2D edgelets
    /// @tparam StatusIt Iterator type of the input statuses
    /// @tparam OutputEdgelet2dIt Iterator type of the output 2D edgelets
    /// @param time_surface The time surface in which 2D edgelets are looked for
    /// @param target Target timestamp used for matching. A timestamp ts in the time surface will match a support point
    /// if ts > (target - threshold_)
    /// @param edgelet_begin First iterator to the buffer of 2D edgelets that will be looked for
    /// @param edgelet_end Last iterator to the buffer of 2D edgelets that will be looked for
    /// @param d_begin First iterator to the matched 2D edgelets buffer
    /// @param status_begin First iterator to the tracking statuses buffer
    /// @return The last iterator to the matched 2D edgelets
    /// @warning The output matched 2D edgelets buffer needs to be resized to only contain the matched edgelets, or a
    /// std::back_insert_iterator needs to be passed instead.
    /// @code{.cpp}
    /// Edgelet2dTrackingAlgorithm algo;
    /// MostRecentTimestampBuffer time_surface(height, width, n_channels);
    /// std::vector<EventEdgelet2d> detected_edgelets;
    /// // Detect edgelets
    /// ...
    /// std::vector<EventEdgelet2d> tracked_edgelets(detected_edgelets.size());
    /// std::vector<bool> statuses(detected_edgelets.size());
    ///
    /// auto tracked_edglet_end = algo.process(time_surface, target_ts, detected_edgelets.cbegin(),
    ///                                        detected_edgelets.cend(), tracked_edgelets.begin(), statuses.begin());
    /// tracked_edgelets.resize(std::distance(tracked_edgelets.end(), tracked_edglet_end));
    /// @endcode
    /// or
    /// @code{.cpp}
    /// Edgelet2dTrackingAlgorithm algo;
    /// MostRecentTimestampBuffer time_surface(height, width, n_channels);
    /// std::vector<EventEdgelet2d> detected_edgelets;
    /// // Detect edgelets
    /// ...
    /// std::vector<EventEdgelet2d> tracked_edgelets;
    /// std::vector<bool> statuses);
    ///
    /// auto tracked_edglet_end = algo.process(time_surface, target_ts, detected_edgelets.cbegin(),
    ///                                        detected_edgelets.cend(),
    ///                                        std::back_insert_iterator(tracked_edgelets.begin()),
    ///                                        std::back_insert_iterator(statuses.begin()));
    /// @endcode
    template<typename Edgelet2dIt, typename StatusIt, typename OutputEdgelet2dIt>
    OutputEdgelet2dIt process(const MostRecentTimestampBuffer &time_surface, timestamp target,
                              Edgelet2dIt edgelet_begin, Edgelet2dIt edgelet_end, OutputEdgelet2dIt d_begin,
                              StatusIt status_begin);

    /// @brief Returns the algorithm's parameters
    const Parameters &get_parameters() const;

private:
    const Parameters params_;
    std::vector<cv::Matx21f> matches_;
    std::vector<int> matches_idx_;
    std::vector<cv::Matx21f> match_candidates_;
};

} // namespace Metavision

#include "metavision/sdk/cv3d/algorithms/detail/edgelet_2d_tracking_algorithm_impl.h"

#endif // METAVISION_SDK_CV3D_EDGELET_2D_TRACKING_ALGORITHM_H
