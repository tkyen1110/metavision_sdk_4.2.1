/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_LINE_PARTICLE_TRACKING_CONFIG_H
#define METAVISION_SDK_ANALYTICS_LINE_PARTICLE_TRACKING_CONFIG_H

#include <stdexcept>

namespace Metavision {

/// @brief Struct representing the parameters used to instantiate a LineParticleTracker inside
/// @ref PsmAlgorithm
struct LineParticleTrackingConfig {
    LineParticleTrackingConfig() = default;

    /// @brief Constructor
    /// @param is_going_down True if the particle is falling, false if it's going upwards
    /// @param dt_first_match_ths Maximum allowed duration to match the 2nd particle of a track
    /// @param tan_angle_ths Tangent of the angle with the vertical beyond which two particles on consecutive lines
    /// can't be matched
    /// @param matching_ths Minimum similarity score in [0,1] needed to match two particles
    LineParticleTrackingConfig(bool is_going_down, unsigned int dt_first_match_ths, double tan_angle_ths = 1.,
                               double matching_ths = 0.5) :
        is_going_down_(is_going_down),
        dt_first_match_ths_(dt_first_match_ths),
        tan_angle_ths_(tan_angle_ths),
        matching_ths_(matching_ths) {
        if (tan_angle_ths_ <= 0)
            throw std::invalid_argument("The tangent threshold must be strictly positive.");
    }
    bool is_going_down_;
    unsigned int dt_first_match_ths_;
    double tan_angle_ths_;
    double matching_ths_;
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_LINE_PARTICLE_TRACKING_CONFIG_H