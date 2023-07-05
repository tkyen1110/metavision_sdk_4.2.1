/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_MODEL_3D_DETECTION_ALGORITHM_IMPL_H
#define METAVISION_SDK_CV_MODEL_3D_DETECTION_ALGORITHM_IMPL_H

#include "metavision/sdk/core/utils/mostrecent_timestamp_buffer.h"

namespace Metavision {

template<typename InputIt>
bool Model3dDetectionAlgorithm::process_events(InputIt it_begin, InputIt it_end, Eigen::Matrix4f &T_c_w,
                                               std::set<size_t> *visible_edges, std::set<size_t> *detected_edges) {
    if (it_begin == it_end) {
        return false;
    }

    for (auto it = it_begin; it != it_end; ++it) {
        time_surface_.at(it->y, it->x, it->p) = it->t;
    }

    if (process_internal(visible_edges, detected_edges)) {
        T_c_w = T_c_w_;
        return true;
    }

    return false;
}

} // namespace Metavision

#endif // METAVISION_SDK_CV_MODEL_3D_DETECTION_ALGORITHM_IMPL_H
