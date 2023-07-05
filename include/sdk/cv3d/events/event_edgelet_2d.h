/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV3D_EVENT_EDGELET_2D_H
#define METAVISION_SDK_CV3D_EVENT_EDGELET_2D_H

#include <opencv2/core.hpp>

namespace Metavision {

/// @brief Small 2D edge portion detected in a time surface
struct EventEdgelet2d {
    cv::Matx21f ctr2_;       ///< 2D point representing the center of the edgelet
    cv::Matx21f unit_dir2_;  ///< 2D vector representing the normalized direction of the edgelet
    cv::Matx21f unit_norm2_; ///< 2D vector representing the normalized normal of the edgelet
};

} // namespace Metavision

#endif // METAVISION_SDK_CV3D_EVENT_EDGELET_2D_H
