/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_DETAIL_MAT_TRAITS_EIGEN_H
#define METAVISION_SDK_CV_DETAIL_MAT_TRAITS_EIGEN_H

#include <Eigen/Core>

#include "metavision/sdk/cv/utils/detail/mat_traits_base.h"

namespace Metavision {

struct static_eigen_matrix_tag {};

struct dynamic_eigen_matrix_tag {};

template<int EIGEN_OPTION>
struct get_storage_order {
    static constexpr StorageOrder type = StorageOrder::NOT_DEFINED;
};

template<>
struct get_storage_order<Eigen::RowMajor> {
    static constexpr StorageOrder type = StorageOrder::ROW_MAJOR;
};

template<>
struct get_storage_order<Eigen::ColMajor> {
    static constexpr StorageOrder type = StorageOrder::COLUMN_MAJOR;
};

template<typename T, int ROWS, int COLS, int EIGEN_OPTION>
struct mat_traits<Eigen::Matrix<T, ROWS, COLS, EIGEN_OPTION>> {
    static constexpr bool is_dynamic = (ROWS == Eigen::Dynamic || COLS == Eigen::Dynamic);

    using value_type = T;
    using category   = typename std::conditional<is_dynamic, dynamic_eigen_matrix_tag, static_eigen_matrix_tag>::type;

    static constexpr StorageOrder storage_order = get_storage_order<EIGEN_OPTION>::type;
    static constexpr int dimX                   = (COLS != -1) ? COLS : 0;
    static constexpr int dimY                   = (ROWS != -1) ? ROWS : 0;
};

template<typename T>
struct mat_traits<Eigen::Ref<T>> {
    using value_type = typename mat_traits<T>::value_type;
    using category   = typename mat_traits<T>::category;

    static constexpr StorageOrder storage_order = mat_traits<T>::storage_order;
    static constexpr int dimX                   = mat_traits<T>::dimX;
    static constexpr int dimY                   = mat_traits<T>::dimY;
};

template<typename T, int MAP_OPTIONS, typename StrideType>
struct mat_traits<Eigen::Map<T, MAP_OPTIONS, StrideType>> {
    using value_type = typename mat_traits<T>::value_type;
    using category   = typename mat_traits<T>::category;

    static constexpr StorageOrder storage_order = mat_traits<T>::storage_order;
    static constexpr int dimX                   = mat_traits<T>::dimX;
    static constexpr int dimY                   = mat_traits<T>::dimY;
};

template<typename T, int ROWS, int COLS, int EIGEN_OPTION>
struct underlying_matrix_type<Eigen::Matrix<T, ROWS, COLS, EIGEN_OPTION>> {
    using type = Eigen::Matrix<T, ROWS, COLS, EIGEN_OPTION>;
};

template<typename T>
struct underlying_matrix_type<Eigen::Map<T>> {
    using type = typename underlying_matrix_type<T>::type;
};

template<typename T>
struct underlying_matrix_type<Eigen::Ref<T>> {
    using type = typename underlying_matrix_type<T>::type;
};

namespace detail {

template<typename T, int ROW, int COL>
inline const typename mat_traits<T>::value_type &get_mat_element_dispatch(const T &mat, static_eigen_matrix_tag) {
    return mat(ROW, COL);
}

template<typename T, int ROW, int COL>
inline typename mat_traits<T>::value_type &get_mat_element_dispatch(T &mat, static_eigen_matrix_tag) {
    return mat(ROW, COL);
}

template<typename T>
inline const typename mat_traits<T>::value_type &get_mat_element_dispatch(const T &mat, int row, int col,
                                                                          dynamic_eigen_matrix_tag) {
    return mat(row, col);
}

template<typename T>
inline typename mat_traits<T>::value_type &get_mat_element_dispatch(T &mat, int row, int col,
                                                                    dynamic_eigen_matrix_tag) {
    return mat(row, col);
}

template<typename T>
inline const typename mat_traits<T>::value_type *get_raw_pointer_dispatch(const T &mat, static_eigen_matrix_tag) {
    return mat.data();
}

template<typename T>
inline typename mat_traits<T>::value_type *get_raw_pointer_dispatch(T &mat, static_eigen_matrix_tag) {
    return mat.data();
}
} // namespace detail
} // namespace Metavision

#endif // METAVISION_SDK_CV_DETAIL_MAT_TRAITS_EIGEN_H