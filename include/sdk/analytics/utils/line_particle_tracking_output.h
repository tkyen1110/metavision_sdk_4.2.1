/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_LINE_PARTICLE_TRACKING_OUTPUT_H
#define METAVISION_SDK_ANALYTICS_LINE_PARTICLE_TRACKING_OUTPUT_H

#include <opencv2/core/types.hpp>
#include <vector>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/analytics/utils/optim_vector.h"

namespace Metavision {

/// @brief Structure storing information about a track of a particle matched over several rows
struct LineParticleTrack {
    /// @brief Default Constructor
    LineParticleTrack() = default;

    /// @brief Move Constructor
    /// @param other Object to move
    LineParticleTrack(LineParticleTrack &&other);

    /// @brief Move assignement operator
    /// @param other Object to move
    LineParticleTrack &operator=(LineParticleTrack &&other);

    void clear();

    std::vector<cv::Point> positions;                        ///< XY Positions of the detections
    std::vector<std::vector<cv::Point2f>> centered_contours; ///< Particle contours centered around (0,0)
    float traj_coef_a, traj_coef_b;                          ///< Linear Model X = a*Y + b
    float particle_size;                                     ///< Estimated size of the particle
    int id;                                                  ///< Track id

    timestamp t; ///< Timestamp
};

/// @brief Class collecting information about LineParticle tracks
struct LineParticleTrackingOutput {
    LineParticleTrackingOutput() = default;

    void clear();

    OptimVector<LineParticleTrack> buffer; ///< Vector of line particle tracks
    timestamp last_count_ts = 0;           ///< Timestamp of the last detection
    int global_counter      = 0;           ///< Number of particles that have been matched over several lines
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_LINE_PARTICLE_TRACKING_OUTPUT_H
