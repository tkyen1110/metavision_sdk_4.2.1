/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV3D_DETAIL_EDGELET_UTILS_IMPL_H
#define METAVISION_SDK_CV3D_DETAIL_EDGELET_UTILS_IMPL_H

#include "metavision/sdk/cv/utils/detail/vec_traits.h"

namespace Metavision {

template<typename Ti, typename To>
inline To edgelet_direction_from_normal(const Ti &normal) {
    To direction;
    get_x_element(direction) = -get_y_element(normal);
    get_y_element(direction) = get_x_element(normal);
    return direction;
}

template<typename Ti, typename To>
inline To edgelet_normal_from_direction(const Ti &direction) {
    To normal;
    get_x_element(normal) = get_y_element(direction);
    get_y_element(normal) = -get_x_element(direction);
    return normal;
}

} // namespace Metavision

#endif // METAVISION_SDK_CV3D_DETAIL_EDGELET_UTILS_IMPL_H
