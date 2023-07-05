/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_DETAIL_MAT_TRAITS_H
#define METAVISION_SDK_CV_DETAIL_MAT_TRAITS_H

#include <type_traits>
#include <opencv2/core.hpp>

#include "metavision/sdk/cv/utils/detail/vec_traits.h"
#include "metavision/sdk/cv/utils/detail/mat_traits_base.h"
#include "metavision/sdk/cv/utils/detail/mat_traits_eigen.h"

namespace Metavision {

struct static_opencv_matrix_tag {};

template<typename T, int ROWS, int COLS>
struct mat_traits<cv::Matx<T, ROWS, COLS>> {
    using value_type                            = T;
    using category                              = static_opencv_matrix_tag;
    static constexpr StorageOrder storage_order = StorageOrder::ROW_MAJOR;
    static constexpr int dimX                   = COLS;
    static constexpr int dimY                   = ROWS;
};

template<typename T>
struct mat_traits<cv::Mat_<T>> {
    using value_type                            = T;
    using category                              = dynamic_opencv_matrix_tag;
    static constexpr StorageOrder storage_order = StorageOrder::ROW_MAJOR;
    static constexpr int dimX                   = 0;
    static constexpr int dimY                   = 0;
};

template<typename T, int ROWS, int COLS>
struct underlying_matrix_type<cv::Matx<T, ROWS, COLS>> {
    using type = cv::Matx<T, ROWS, COLS>;
};

namespace detail {

template<typename T, int ROW, int COL>
inline const typename mat_traits<T>::value_type &get_mat_element_dispatch(const T &mat, static_opencv_matrix_tag) {
    return mat(ROW, COL);
}

template<typename T, int ROW, int COL>
inline typename mat_traits<T>::value_type &get_mat_element_dispatch(T &mat, static_opencv_matrix_tag) {
    return mat(ROW, COL);
}

template<typename T>
inline const typename mat_traits<T>::value_type &get_mat_element_dispatch(const T &mat, int row, int col,
                                                                          dynamic_opencv_matrix_tag) {
    return mat(row, col);
}

template<typename T>
inline typename mat_traits<T>::value_type &get_mat_element_dispatch(T &mat, int row, int col,
                                                                    dynamic_opencv_matrix_tag) {
    return mat(row, col);
}

template<typename T>
inline const typename mat_traits<T>::value_type *get_raw_pointer_dispatch(const T &mat, static_opencv_matrix_tag) {
    return mat.val;
}

template<typename T>
inline typename mat_traits<T>::value_type *get_raw_pointer_dispatch(T &mat, static_opencv_matrix_tag) {
    return mat.val;
}

template<typename T>
inline const typename mat_traits<T>::value_type *get_raw_pointer_dispatch(const T &mat, dynamic_opencv_matrix_tag) {
    return reinterpret_cast<const typename mat_traits<T>::value_type *>(mat.data);
}

template<typename T>
inline typename mat_traits<T>::value_type *get_raw_pointer_dispatch(T &mat, dynamic_opencv_matrix_tag) {
    return reinterpret_cast<typename mat_traits<T>::value_type *>(mat.data);
}

} // namespace detail

template<typename M, int ROWS, int COLS>
struct is_valid_matrix {
    static constexpr bool is_underlying_type_valid = !std::is_void<typename mat_traits<M>::value_type>::value;
    static constexpr bool are_dimensions_valid     = (mat_traits<M>::dimY == 0 || mat_traits<M>::dimY >= ROWS) &&
                                                 (mat_traits<M>::dimX == 0 || mat_traits<M>::dimX >= COLS);

    static constexpr bool value = (is_underlying_type_valid && are_dimensions_valid);
};

template<typename M>
struct is_valid_dynamic_matrix {
    static constexpr bool value = is_valid_matrix<M, 0, 0>::value;
};

// Definition of the generic getter/setter functions
template<typename T, int ROW, int COL>
inline const typename mat_traits<T>::value_type &get_mat_element(const T &mat) {
    static_assert(!std::is_void<typename mat_traits<T>::value_type>::value, "Unsupported matrix type!");
    static_assert(mat_traits<T>::dimX > COL, "Incorrect matrix's X dimension!");
    static_assert(mat_traits<T>::dimY > ROW, "Incorrect matrix's Y dimension!");
    typename mat_traits<T>::category c;
    return detail::get_mat_element_dispatch<T, ROW, COL>(mat, c);
}

template<typename T, int ROW, int COL>
inline typename mat_traits<T>::value_type &get_mat_element(T &mat) {
    static_assert(!std::is_void<typename mat_traits<T>::value_type>::value, "Unsupported matrix type!");
    static_assert(mat_traits<T>::dimX > COL, "Incorrect matrix's X dimension!");
    static_assert(mat_traits<T>::dimY > ROW, "Incorrect matrix's Y dimension!");
    typename mat_traits<T>::category c;
    return detail::get_mat_element_dispatch<T, ROW, COL>(mat, c);
}

template<typename T>
inline const typename mat_traits<T>::value_type &get_mat_element(const T &mat, int row, int col) {
    static_assert(!std::is_void<typename mat_traits<T>::value_type>::value, "Unsupported matrix type!");
    typename mat_traits<T>::category c;
    return detail::get_mat_element_dispatch<T>(mat, row, col, c);
}

template<typename T>
inline typename mat_traits<T>::value_type &get_mat_element(T &mat, int row, int col) {
    static_assert(!std::is_void<typename mat_traits<T>::value_type>::value, "Unsupported matrix type!");
    typename mat_traits<T>::category c;
    return detail::get_mat_element_dispatch<T>(mat, row, col, c);
}

template<typename T>
inline const typename mat_traits<T>::value_type *get_mat_raw_ptr(const T &mat) {
    static_assert(!std::is_void<typename mat_traits<T>::value_type>::value, "Unsupported matrix type!");
    typename mat_traits<T>::category c;
    return detail::get_raw_pointer_dispatch(mat, c);
}

template<typename T>
inline typename mat_traits<T>::value_type *get_mat_raw_ptr(T &mat) {
    static_assert(!std::is_void<typename mat_traits<T>::value_type>::value, "Unsupported matrix type!");
    typename mat_traits<T>::category c;
    return detail::get_raw_pointer_dispatch(mat, c);
}

} // namespace Metavision

#endif // METAVISION_SDK_CV_DETAIL_MAT_TRAITS_H