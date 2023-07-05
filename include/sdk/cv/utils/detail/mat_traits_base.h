/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_DETAIL_MAT_TRAITS_BASE_H
#define METAVISION_SDK_CV_DETAIL_MAT_TRAITS_BASE_H

namespace Metavision {

/// @brief Enumerate telling how matrix' data is organized in memory
enum class StorageOrder { ROW_MAJOR, COLUMN_MAJOR, NOT_DEFINED };

/// @brief Trait structure used to retrieve information about a matrix type
///
/// The default structure implementation will be used for types corresponding to invalid (or unsupported) matrix
/// representations, whereas the specialized structure implementations will enable to distinguish between the different
/// supported matrix representations.
///
/// @tparam T Type corresponding to a matrix
template<typename T>
struct mat_traits {
    using value_type                            = void;                      ///< Type of the matrix values
    using category                              = void;                      ///< The Matrix category
    static constexpr StorageOrder storage_order = StorageOrder::NOT_DEFINED; ///< Storage layout in memory
    static constexpr int dimX                   = -1;                        ///< The number of columns in the matrix
    static constexpr int dimY                   = -1;                        ///< The number of rows in the matrix
};

template<typename T>
struct underlying_matrix_type;

} // namespace Metavision

#endif // METAVISION_SDK_CV_DETAIL_MAT_TRAITS_BASE_H