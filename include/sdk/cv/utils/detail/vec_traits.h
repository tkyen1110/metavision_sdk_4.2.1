/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_DETAIL_VEC_TRAITS_H
#define METAVISION_SDK_CV_DETAIL_VEC_TRAITS_H

#include <type_traits>
#include <vector>
#include <array>
#include <opencv2/core/core.hpp>

#include "metavision/sdk/cv/utils/detail/vec_traits_base.h"
#include "metavision/sdk/cv/utils/detail/vec_traits_eigen.h"

namespace Metavision {

struct random_access_vector_tag {};
struct dynamic_opencv_matrix_tag {};
struct static_opencv_point_tag {};
struct static_opencv_vector_tag {};

template<typename T>
struct vec_traits<T *> {
    typedef T value_type;
    typedef random_access_vector_tag category;
    enum { dim = 0 };
};

template<typename T>
struct vec_traits<std::vector<T>> {
    typedef T value_type;
    typedef random_access_vector_tag category;
    enum { dim = 0 };
};

template<typename T, int ROWS>
struct vec_traits<std::array<T, ROWS>> {
    typedef T value_type;
    typedef random_access_vector_tag category;
    enum { dim = ROWS };
};

template<typename T>
struct vec_traits<cv::Mat_<T>> {
    typedef T value_type;
    typedef dynamic_opencv_matrix_tag category;
    enum { dim = 0 };
};

template<typename T>
struct vec_traits<cv::Point_<T>> {
    typedef T value_type;
    typedef static_opencv_point_tag category;
    enum { dim = 2 };
};

template<typename T>
struct vec_traits<cv::Point3_<T>> {
    typedef T value_type;
    typedef static_opencv_point_tag category;
    enum { dim = 3 };
};

template<typename T, int ROWS>
struct vec_traits<cv::Vec<T, ROWS>> {
    typedef T value_type;
    typedef static_opencv_vector_tag category;
    enum { dim = ROWS };
};

template<typename T, int ROWS>
struct vec_traits<cv::Matx<T, ROWS, 1>> {
    typedef T value_type;
    typedef static_opencv_vector_tag category;
    enum { dim = ROWS };
};

template<typename T, int COLS>
struct vec_traits<cv::Matx<T, 1, COLS>> {
    typedef T value_type;
    typedef static_opencv_vector_tag category;
    enum { dim = COLS };
};

/// Implementation of the getter/setter functions specialized for each way of element access.
namespace detail {

template<typename T>
inline const typename vec_traits<T>::value_type &get_x_element_dispatch(const T &vec, random_access_vector_tag) {
    return vec[0];
}
template<typename T>
inline typename vec_traits<T>::value_type &get_x_element_dispatch(T &vec, random_access_vector_tag) {
    return vec[0];
}
template<typename T>
inline const typename vec_traits<T>::value_type &get_x_element_dispatch(const T &vec, dynamic_opencv_matrix_tag) {
    return vec.template at<typename vec_traits<T>::value_type>(0);
}
template<typename T>
inline typename vec_traits<T>::value_type &get_x_element_dispatch(T &vec, dynamic_opencv_matrix_tag) {
    return vec.template at<typename vec_traits<T>::value_type>(0);
}
template<typename T>
inline const typename vec_traits<T>::value_type &get_x_element_dispatch(const T &vec, static_opencv_point_tag) {
    return vec.x;
}
template<typename T>
inline typename vec_traits<T>::value_type &get_x_element_dispatch(T &vec, static_opencv_point_tag) {
    return vec.x;
}
template<typename T>
inline const typename vec_traits<T>::value_type &get_x_element_dispatch(const T &vec, static_opencv_vector_tag) {
    return vec.val[0];
}
template<typename T>
inline typename vec_traits<T>::value_type &get_x_element_dispatch(T &vec, static_opencv_vector_tag) {
    return vec.val[0];
}

template<typename T>
inline const typename vec_traits<T>::value_type &get_y_element_dispatch(const T &vec, random_access_vector_tag) {
    return vec[1];
}
template<typename T>
inline typename vec_traits<T>::value_type &get_y_element_dispatch(T &vec, random_access_vector_tag) {
    return vec[1];
}
template<typename T>
inline const typename vec_traits<T>::value_type &get_y_element_dispatch(const T &vec, dynamic_opencv_matrix_tag) {
    return vec.template at<typename vec_traits<T>::value_type>(1);
}
template<typename T>
inline typename vec_traits<T>::value_type &get_y_element_dispatch(T &vec, dynamic_opencv_matrix_tag) {
    return vec.template at<typename vec_traits<T>::value_type>(1);
}
template<typename T>
inline const typename vec_traits<T>::value_type &get_y_element_dispatch(const T &vec, static_opencv_point_tag) {
    return vec.y;
}
template<typename T>
inline typename vec_traits<T>::value_type &get_y_element_dispatch(T &vec, static_opencv_point_tag) {
    return vec.y;
}
template<typename T>
inline const typename vec_traits<T>::value_type &get_y_element_dispatch(const T &vec, static_opencv_vector_tag) {
    return vec.val[1];
}
template<typename T>
inline typename vec_traits<T>::value_type &get_y_element_dispatch(T &vec, static_opencv_vector_tag) {
    return vec.val[1];
}

template<typename T>
inline const typename vec_traits<T>::value_type &get_z_element_dispatch(const T &vec, random_access_vector_tag) {
    return vec[2];
}
template<typename T>
inline typename vec_traits<T>::value_type &get_z_element_dispatch(T &vec, random_access_vector_tag) {
    return vec[2];
}
template<typename T>
inline const typename vec_traits<T>::value_type &get_z_element_dispatch(const T &vec, dynamic_opencv_matrix_tag) {
    return vec.template at<typename vec_traits<T>::value_type>(2);
}
template<typename T>
inline typename vec_traits<T>::value_type &get_z_element_dispatch(T &vec, dynamic_opencv_matrix_tag) {
    return vec.template at<typename vec_traits<T>::value_type>(2);
}
template<typename T>
inline const typename vec_traits<T>::value_type &get_z_element_dispatch(const T &vec, static_opencv_point_tag) {
    return vec.z;
}
template<typename T>
inline typename vec_traits<T>::value_type &get_z_element_dispatch(T &vec, static_opencv_point_tag) {
    return vec.z;
}
template<typename T>
inline const typename vec_traits<T>::value_type &get_z_element_dispatch(const T &vec, static_opencv_vector_tag) {
    return vec.val[2];
}
template<typename T>
inline typename vec_traits<T>::value_type &get_z_element_dispatch(T &vec, static_opencv_vector_tag) {
    return vec.val[2];
}

template<typename T>
inline const typename vec_traits<T>::value_type *get_raw_ptr_dispatch(const T &vec, random_access_vector_tag) {
    return vec.data();
}

template<typename T>
inline typename vec_traits<T>::value_type *get_raw_ptr_dispatch(T &vec, random_access_vector_tag) {
    return vec.data();
}

template<typename T>
inline const typename vec_traits<T>::value_type *get_raw_ptr_dispatch(const T &vec, dynamic_opencv_matrix_tag) {
    return vec.ptr();
}

template<typename T>
inline typename vec_traits<T>::value_type *get_raw_ptr_dispatch(T &vec, dynamic_opencv_matrix_tag) {
    return vec.ptr();
}

template<typename T>
inline const typename vec_traits<T>::value_type *get_raw_ptr_dispatch(const T &vec, static_opencv_point_tag) {
    return &vec.x;
}
template<typename T>
inline typename vec_traits<T>::value_type *get_raw_ptr_dispatch(T &vec, static_opencv_point_tag) {
    return &vec.x;
}

template<typename T>
inline const typename vec_traits<T>::value_type *get_raw_ptr_dispatch(const T &vec, static_opencv_vector_tag) {
    return vec.val;
}

template<typename T>
inline typename vec_traits<T>::value_type *get_raw_ptr_dispatch(T &vec, static_opencv_vector_tag) {
    return vec.val;
}
} // namespace detail

/// Definition of the generic getter/setter functions

/// @brief Trait structure used to check if a vector type is valid
/// @tparam V A vector type
/// @tparam ROWS The expected vector's rows number
template<typename V, int ROWS>
struct is_valid_vector {
    static constexpr bool is_underlying_type_valid = !std::is_void<typename vec_traits<V>::value_type>::value;
    static constexpr bool is_dimension_valid       = (vec_traits<V>::dim == 0 || vec_traits<V>::dim >= ROWS);
    static constexpr bool value                    = (is_underlying_type_valid && is_dimension_valid);
};

template<typename T>
inline const typename vec_traits<T>::value_type &get_x_element(const T &vec) {
    static_assert(!std::is_void<typename vec_traits<T>::value_type>::value, "Unsupported vector type!");
    static_assert(vec_traits<T>::dim == 0 || vec_traits<T>::dim >= 1, "Input vector has no X component!");
    typename vec_traits<T>::category c;
    return detail::get_x_element_dispatch(vec, c);
}
template<typename T>
inline typename vec_traits<T>::value_type &get_x_element(T &vec) {
    static_assert(!std::is_void<typename vec_traits<T>::value_type>::value, "Unsupported vector type!");
    static_assert(vec_traits<T>::dim == 0 || vec_traits<T>::dim >= 1, "Input vector has no X component!");
    typename vec_traits<T>::category c;
    return detail::get_x_element_dispatch(vec, c);
}

template<typename T>
inline const typename vec_traits<T>::value_type &get_y_element(const T &vec) {
    static_assert(!std::is_void<typename vec_traits<T>::value_type>::value, "Unsupported vector type!");
    static_assert(vec_traits<T>::dim == 0 || vec_traits<T>::dim >= 2, "Input vector has no Y component!");
    typename vec_traits<T>::category c;
    return detail::get_y_element_dispatch(vec, c);
}
template<typename T>
inline typename vec_traits<T>::value_type &get_y_element(T &vec) {
    static_assert(!std::is_void<typename vec_traits<T>::value_type>::value, "Unsupported vector type!");
    static_assert(vec_traits<T>::dim == 0 || vec_traits<T>::dim >= 2, "Input vector has no Y component!");
    typename vec_traits<T>::category c;
    return detail::get_y_element_dispatch(vec, c);
}

template<typename T>
inline const typename vec_traits<T>::value_type &get_z_element(const T &vec) {
    static_assert(!std::is_void<typename vec_traits<T>::value_type>::value, "Unsupported vector type!");
    static_assert(vec_traits<T>::dim == 0 || vec_traits<T>::dim >= 3, "Input vector has no Z component!");
    typename vec_traits<T>::category c;
    return detail::get_z_element_dispatch(vec, c);
}
template<typename T>
inline typename vec_traits<T>::value_type &get_z_element(T &vec) {
    static_assert(!std::is_void<typename vec_traits<T>::value_type>::value, "Unsupported vector type!");
    static_assert(vec_traits<T>::dim == 0 || vec_traits<T>::dim >= 3, "Input vector has no Z component!");
    typename vec_traits<T>::category c;
    return detail::get_z_element_dispatch(vec, c);
}

template<typename T>
inline const typename vec_traits<T>::value_type *get_raw_ptr(const T &vec) {
    static_assert(!std::is_void<typename vec_traits<T>::value_type>::value, "Unsupported vector type!");
    typename vec_traits<T>::category c;
    return detail::get_raw_ptr_dispatch(vec, c);
}

template<typename T>
inline typename vec_traits<T>::value_type *get_raw_ptr(T &vec) {
    static_assert(!std::is_void<typename vec_traits<T>::value_type>::value, "Unsupported vector type!");
    typename vec_traits<T>::category c;
    return detail::get_raw_ptr_dispatch(vec, c);
}

} // namespace Metavision

#endif // METAVISION_SDK_CV_DETAIL_VEC_TRAITS_H
