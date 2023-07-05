/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_DETAIL_VEC_TRAITS_EIGEN_H
#define METAVISION_SDK_CV_DETAIL_VEC_TRAITS_EIGEN_H

#include <Eigen/Core>

#include "metavision/sdk/cv/utils/detail/vec_traits_base.h"

namespace Metavision {

struct static_eigen_vector_tag {};
struct static_eigen_reference_tag {};

template<typename T, int ROWS>
struct vec_traits<Eigen::Matrix<T, ROWS, 1>> {
    typedef T value_type;
    typedef static_eigen_vector_tag category;
    enum { dim = ROWS };
};

template<typename T, int options, typename S>
struct vec_traits<Eigen::Ref<T, options, S>> {
    typedef typename Eigen::Ref<T>::Scalar value_type;
    typedef static_eigen_reference_tag category;
    enum { dim = Eigen::Ref<T>::RowsAtCompileTime };
};

/// Implementation of the getter/setter functions specialized for each way of element access.

namespace detail {
template<typename T>
inline const typename vec_traits<T>::value_type &get_x_element_dispatch(const T &vec, static_eigen_vector_tag) {
    return vec(0);
}
template<typename T>
inline typename vec_traits<T>::value_type &get_x_element_dispatch(T &vec, static_eigen_vector_tag) {
    return vec(0);
}

template<typename T>
inline const typename vec_traits<T>::value_type &get_x_element_dispatch(const T &vec, static_eigen_reference_tag) {
    // Using the () operator should work but the code doesn't compile...
    return vec.data()[0];
}
template<typename T>
inline typename vec_traits<T>::value_type &get_x_element_dispatch(T &vec, static_eigen_reference_tag) {
    // Using the () operator should work but the code doesn't compile...
    return vec.data()[0];
}

template<typename T>
inline const typename vec_traits<T>::value_type &get_y_element_dispatch(const T &vec, static_eigen_vector_tag) {
    return vec(1);
}
template<typename T>
inline typename vec_traits<T>::value_type &get_y_element_dispatch(T &vec, static_eigen_vector_tag) {
    return vec(1);
}

template<typename T>
inline const typename vec_traits<T>::value_type &get_y_element_dispatch(const T &vec, static_eigen_reference_tag) {
    // Using the () operator should work but the code doesn't compile...
    return vec.data()[1];
}
template<typename T>
inline typename vec_traits<T>::value_type &get_y_element_dispatch(T &vec, static_eigen_reference_tag) {
    // Using the () operator should work but the code doesn't compile...
    return vec.data()[1];
}

template<typename T>
inline const typename vec_traits<T>::value_type &get_z_element_dispatch(const T &vec, static_eigen_vector_tag) {
    return vec(2);
}
template<typename T>
inline typename vec_traits<T>::value_type &get_z_element_dispatch(T &vec, static_eigen_vector_tag) {
    return vec(2);
}

template<typename T>
inline const typename vec_traits<T>::value_type &get_z_element_dispatch(const T &vec, static_eigen_reference_tag) {
    // Using the () operator should work but the code doesn't compile...
    return vec.data()[2];
}
template<typename T>
inline typename vec_traits<T>::value_type &get_z_element_dispatch(T &vec, static_eigen_reference_tag) {
    // Using the () operator should work but the code doesn't compile...
    return vec.data()[2];
}

template<typename T>
inline const typename vec_traits<T>::value_type *get_raw_ptr_dispatch(const T &vec, static_eigen_vector_tag) {
    return vec.data();
}

template<typename T>
inline typename vec_traits<T>::value_type *get_raw_ptr_dispatch(T &vec, static_eigen_vector_tag) {
    return vec.data();
}

template<typename T>
inline const typename vec_traits<T>::value_type *get_raw_ptr_dispatch(const T &vec, static_eigen_reference_tag) {
    return vec.data();
}

template<typename T>
inline typename vec_traits<T>::value_type *get_raw_ptr_dispatch(T &vec, static_eigen_reference_tag) {
    return vec.data();
}
} // namespace detail
} // namespace Metavision

#endif // METAVISION_SDK_CV_DETAIL_VEC_TRAITS_EIGEN_H
