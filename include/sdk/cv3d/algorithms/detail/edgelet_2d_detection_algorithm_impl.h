/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV3D_DETAIL_EDGELET_2D_DETECTION_ALGORITHM_IMPL_H
#define METAVISION_SDK_CV3D_DETAIL_EDGELET_2D_DETECTION_ALGORITHM_IMPL_H

#include "metavision/sdk/cv3d/utils/edgelet_utils.h"

namespace Metavision {

template<typename InputIt, typename OutputIt>
OutputIt Edgelet2dDetectionAlgorithm::process(const MostRecentTimestampBuffer &ts, InputIt begin, InputIt end,
                                              OutputIt d_begin) {
    cv::Matx21f ctr2, unit_dir2, unit_norm2;
    for (auto it = begin; it != end; ++it) {
        if (is_fast_edge(ts, *it, threshold_, &unit_norm2)) {
            ctr2       = cv::Matx21f(it->x, it->y);
            unit_dir2  = edgelet_direction_from_normal(unit_norm2);
            *d_begin++ = {ctr2, unit_dir2, unit_norm2};
        }
    }

    return d_begin;
}
} // namespace Metavision

#endif // METAVISION_SDK_CV3D_DETAIL_EDGELET_2D_DETECTION_ALGORITHM_IMPL_H
