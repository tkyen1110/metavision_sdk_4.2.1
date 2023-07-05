/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_DETAIL_VEC_TRAITS_BASE_H
#define METAVISION_SDK_CV_DETAIL_VEC_TRAITS_BASE_H

namespace Metavision {

/// @brief Trait structure used to retrieve information about a vector type
///
/// The default structure implementation will be used for types corresponding to invalid (or unsupported) vector
/// representations, whereas the specialized structure implementations will enable to distinguish between the different
/// supported vector representations.
template<typename T>
struct vec_traits {
    typedef void value_type; ///< Type of the vector values
    typedef void category;   ///< Category of the vector
    enum { dim = -1 };       ///< Dimension of the vector when known at compile-time
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_DETAIL_VEC_TRAITS_BASE_H
