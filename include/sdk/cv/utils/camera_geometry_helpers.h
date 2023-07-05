/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_CAMERA_GEOMETRY_HELPERS_H
#define METAVISION_SDK_CV_CAMERA_GEOMETRY_HELPERS_H

#include <iostream>
#include <Eigen/Dense>

#include "metavision/sdk/cv/utils/detail/vec_traits.h"
#include "metavision/sdk/cv/utils/detail/mat_traits.h"
#include "metavision/sdk/cv/utils/camera_geometry_base.h"

namespace Metavision {
namespace detail {

template<typename T>
struct is_row_major {
    static constexpr bool value = (mat_traits<T>::storage_order == StorageOrder::ROW_MAJOR);
};
} // namespace detail

template<typename T>
class CameraGeometryBase;

/// @brief Tests if a point is visible in the image
/// @tparam T The floating point type used
/// @tparam T1 The 2D point type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the point
/// @param[in] pt_img The point's coordinates in the distorted image plane
/// @param[in] border_margin An optional border margin
/// @return true if the point is visible, false otherwise
template<typename T, typename T1>
inline bool is_visible(const CameraGeometryBase<T> &camera_geometry, const T1 &pt_img, int border_margin = 0) {
    static_assert(is_valid_vector<T1, 2>::value,
                  "Invalid input point type or wrong dimensions. Must have at least 2 dimensions!");

    const auto &x        = get_x_element(pt_img);
    const auto &y        = get_y_element(pt_img);
    const auto &img_size = camera_geometry.get_image_size();
    const auto &width    = img_size(0);
    const auto &height   = img_size(1);

    if (x < border_margin || x > width - 1 - border_margin || y < border_margin || y > height - 1 - border_margin)
        return false;

    return true;
}

/// @brief Normalizes a 3D vector
/// @tparam V The 3D vector type used
/// @param[in,out] vec The vector to normalize
/// @param[in] eps To be normalized the vector's norm should be greater than this value
/// @return true if the vector has been normalized, false otherwise
template<typename V>
inline std::enable_if_t<vec_traits<V>::dim == 3, bool>
    normalize_vector(V &vec, typename vec_traits<V>::value_type eps = typename vec_traits<V>::value_type(1e-10)) {
    static_assert(!std::is_void<typename vec_traits<V>::value_type>::value, "Unsupported vector type!");

    using T       = typename vec_traits<V>::value_type;
    const T norm2 = get_x_element(vec) * get_x_element(vec) + get_y_element(vec) * get_y_element(vec) +
                    get_z_element(vec) * get_z_element(vec);

    if (norm2 < T(eps * eps)) {
        return false;
    }

    T norm_inv = T(1) / std::sqrt(norm2);
    get_x_element(vec) *= norm_inv;
    get_y_element(vec) *= norm_inv;
    get_z_element(vec) *= norm_inv;

    return true;
}

/// @brief Normalizes a 2D vector
/// @tparam V The 2D vector type used
/// @param[in,out] vec The vector to normalize
/// @param[in] eps To be normalized the vector's norm should be greater than this value
/// @return true if the vector has been normalized, false otherwise
template<typename V>
inline std::enable_if_t<vec_traits<V>::dim == 2, bool>
    normalize_vector(V &vec, typename vec_traits<V>::value_type eps = typename vec_traits<V>::value_type(1e-10)) {
    static_assert(!std::is_void<typename vec_traits<V>::value_type>::value, "Unsupported vector type!");

    using T       = typename vec_traits<V>::value_type;
    const T norm2 = get_x_element(vec) * get_x_element(vec) + get_y_element(vec) * get_y_element(vec);

    if (norm2 < T(eps * eps)) {
        return false;
    }

    T norm_inv = T(1) / std::sqrt(norm2);
    get_x_element(vec) *= norm_inv;
    get_y_element(vec) *= norm_inv;

    return true;
}

/// @brief Maps a point from the undistorted normalized image plane into the camera's coordinates system
/// @tparam T The floating point type used
/// @tparam Ti The input 2D point type used
/// @tparam To The output 3D point type used
/// @param[in] pt_norm The point in the undistorted normalized image plane
/// @param[out] pt_c The point in the camera's coordinates system
/// @param[in] depth The depth of the point in the camera's coordinates system
template<typename T, typename Ti, typename To>
inline void undist_norm_to_camera(const Ti &pt_norm, To &pt_c, T depth) {
    static_assert(is_valid_vector<Ti, 2>::value,
                  "Invalid input point type or wrong dimensions. Must have at least 2 dimensions!");
    static_assert(is_valid_vector<To, 3>::value,
                  "Invalid output point type or wrong dimensions. Must have at least 3 dimensions!");

    get_x_element(pt_c) = depth * get_x_element(pt_norm);
    get_y_element(pt_c) = depth * get_y_element(pt_norm);
    get_z_element(pt_c) = depth;
}

/// @brief Maps a 3D point from the camera's coordinates system into the undistorted normalized image plane
/// @tparam Ti The input 3D point type used
/// @tparam To The output 2D point type used
/// @param[in] pt3_cam The 3D point expressed in the camera's coordinate system
/// @param[out] pt2_norm The point in the undistorted normalized image plane
template<typename Ti, typename To>
inline void camera_to_undist_norm(const Ti &pt3_cam, To &pt2_norm) {
    static_assert(is_valid_vector<Ti, 3>::value,
                  "Invalid input point type or wrong dimensions. Must have at least 3 dimensions!");
    static_assert(is_valid_vector<To, 2>::value,
                  "Invalid output point type or wrong dimensions. Must have at least 2 dimensions!");

    using T = typename vec_traits<Ti>::value_type;

    const T invz            = 1 / T(get_z_element(pt3_cam));
    get_x_element(pt2_norm) = get_x_element(pt3_cam) * invz;
    get_y_element(pt2_norm) = get_y_element(pt3_cam) * invz;
}

/// @brief Maps a 3D vector from the camera's coordinates system into the undistorted normalized image plane
/// @tparam Ti The input 3D vector type used
/// @tparam To The output 2D vector type used
/// @param[in] v3_cam The 3D vector in the camera's coordinates system
/// @param[in] pt3_cam The vector's starting point in the camera's coordinates system
/// @param[out] v2_undist_norm The vector in the undistorted normalized image plane
/// @param[out] pt2_undist_norm The vector's starting point in the normalized image plane
/// @param normalize The output is normalized (i.e. norm = 1) if this parameter is set to true
template<typename Ti, typename To>
inline void vector_camera_to_undist_norm(const Ti &v3_cam, const Ti &pt3_cam, To &v2_undist_norm, To &pt2_undist_norm,
                                         bool normalize = true) {
    static_assert(is_valid_vector<Ti, 3>::value,
                  "Invalid input vector type or wrong dimensions. Must have at least 3 dimensions!");
    static_assert(is_valid_vector<To, 2>::value,
                  "Invalid output vector type or wrong dimensions. Must have at least 2 dimensions!");

    using T = typename vec_traits<Ti>::value_type;

    T inv_d = 1 / get_z_element(pt3_cam), inv_d2 = inv_d * inv_d;
    T gradx_d                      = -inv_d2 * get_x_element(pt3_cam);
    T grady_d                      = -inv_d2 * get_y_element(pt3_cam);
    get_x_element(v2_undist_norm)  = inv_d * get_x_element(v3_cam) + gradx_d * get_z_element(v3_cam);
    get_y_element(v2_undist_norm)  = inv_d * get_y_element(v3_cam) + grady_d * get_z_element(v3_cam);
    get_x_element(pt2_undist_norm) = get_x_element(pt3_cam) * inv_d;
    get_y_element(pt2_undist_norm) = get_y_element(pt3_cam) * inv_d;

    if (normalize)
        normalize_vector(v2_undist_norm);
}

/// @brief Gets the transform that maps a point from the undistorted normalized image plane (i.e. Z = 1) into the
/// undistorted image plane
/// @tparam T The floating point type used
/// @tparam To The 3x3 matrix type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the point
/// @param[out] m The transform
template<typename T, typename To>
inline void get_undist_norm_to_undist_img_transform(const CameraGeometryBase<T> &camera_geometry, To &m) {
    static_assert(is_valid_matrix<To, 3, 3>::value,
                  "Invalid output matrix type or wrong dimensions. Must be at least of dimensions 3x3!");

    using Base    = CameraGeometryBase<T>;
    using Mat3    = std::conditional_t<detail::is_row_major<To>::value, typename Base::Mat3RM, typename Base::Mat3CM>;
    using Mat3Map = Eigen::Map<Mat3>;

    camera_geometry.get_undist_norm_to_undist_img_transform(Mat3Map(get_mat_raw_ptr(m)));
}

/// @brief Maps a point from the camera's coordinates system into the distorted image plane
/// @tparam T The floating point type used
/// @tparam T1 The 3D point type used
/// @tparam T2 The 2D point type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the point
/// @param[in] pt_c The 3D point in the camera's coordinates system
/// @param[out] pt_dist_img The mapped point in the distorted image plane
template<typename T, typename T1, typename T2>
inline void camera_to_img(const CameraGeometryBase<T> &camera_geometry, const T1 &pt_c, T2 &pt_dist_img) {
    static_assert(is_valid_vector<T1, 3>::value,
                  "Invalid input point type or wrong dimensions. Must have at least 3 dimensions!");
    static_assert(is_valid_vector<T2, 2>::value,
                  "Invalid output point type or wrong dimensions. Must have at least 2 dimensions!");

    using Base    = CameraGeometryBase<T>;
    using Vec2Map = Eigen::Map<typename Base::Vec2>;
    using Vec3Map = Eigen::Map<const typename Base::Vec3>;

    camera_geometry.camera_to_img(Vec3Map(get_raw_ptr(pt_c)), Vec2Map(get_raw_ptr(pt_dist_img)));
}

/// @brief Maps a point from the camera's coordinates system into the distorted image plane
/// @tparam T The floating point type used
/// @tparam T1 The 3D point type used
/// @tparam T2 The 2D point type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the point
/// @param[in] pt_c The 3D point in the camera's coordinates system
/// @param[out] pt_dist_img The mapped point in the distorted image plane
/// @param[in] epsilonz To be mapped, the point's depth in the camera's coordinates system must less than this value
/// @return true if the point has been successfully mapped, false otherwise
template<typename T, typename T1, typename T2>
inline bool safe_camera_to_img(const CameraGeometryBase<T> &camera_geometry, const T1 &pt_c, T2 &pt_dist_img,
                               T epsilonz = T(1e-10)) {
    static_assert(is_valid_vector<T1, 3>::value,
                  "Invalid input point type or wrong dimensions. Must have at least 3 dimensions!");
    static_assert(is_valid_vector<T2, 2>::value,
                  "Invalid output point type or wrong dimensions. Must have at least 2 dimensions!");

    if (std::abs(get_z_element(pt_c)) < epsilonz)
        return false;

    camera_to_img(camera_geometry, pt_c, pt_dist_img);

    return true;
}

/// @brief Maps a point from the camera's coordinates system into the undistorted image plane
/// @tparam T The floating point type used
/// @tparam T1 The 3D point type used
/// @tparam T2 The 2D point type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the point
/// @param[in] pt_c The 3D point in the camera's coordinates system
/// @param[out] pt_undist_img The mapped point in the undistorted image plane
template<typename T, typename T1, typename T2>
inline void camera_to_undist_img(const CameraGeometryBase<T> &camera_geometry, const T1 &pt_c, T2 &pt_undist_img) {
    static_assert(is_valid_vector<T1, 3>::value,
                  "Invalid input point type or wrong dimensions. Must have at least 3 dimensions!");
    static_assert(is_valid_vector<T2, 2>::value,
                  "Invalid output point type or wrong dimensions. Must have at least 2 dimensions!");

    using Base    = CameraGeometryBase<T>;
    using Vec2Map = Eigen::Map<typename Base::Vec2>;
    using Vec3Map = Eigen::Map<const typename Base::Vec3>;

    camera_geometry.camera_to_undist_img(Vec3Map(get_raw_ptr(pt_c)), Vec2Map(get_raw_ptr(pt_undist_img)));
}

/// @brief Maps a point from the undistorted normalized image plane into the undistorted image plane
/// @tparam T The floating point type used
/// @tparam T1 The input 2D point type used
/// @tparam T2 The output 2D point type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the point
/// @param[in] pt_undist_norm The point in the undistorted normalized image plane
/// @param[out] pt_undist_img The mapped point in the undistorted image plane
template<typename T, typename T1, typename T2>
inline void undist_norm_to_undist_img(const CameraGeometryBase<T> &camera_geometry, const T1 &pt_undist_norm,
                                      T2 &pt_undist_img) {
    static_assert(is_valid_vector<T1, 2>::value,
                  "Invalid input point type or wrong dimensions. Must have at least 2 dimensions!");
    static_assert(is_valid_vector<T2, 2>::value,
                  "Invalid output point type or wrong dimensions. Must have at least 2 dimensions!");

    using Base         = CameraGeometryBase<T>;
    using Vec2         = typename Base::Vec2;
    using Vec2Map      = Eigen::Map<Vec2>;
    using Vec2MapConst = Eigen::Map<const Vec2>;

    camera_geometry.undist_norm_to_undist_img(Vec2MapConst(get_raw_ptr(pt_undist_norm)),
                                              Vec2Map(get_raw_ptr(pt_undist_img)));
}

/// @brief Maps a point from the undistorted image plane into the undistorted normalized image plane
/// @tparam T The floating point type used
/// @tparam T1 The input 2D point type used
/// @tparam T2 The output 2D point type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the point
/// @param[in] pt_undist_img The point in the undistorted image plane
/// @param[out] pt_undist_norm The mapped point in the undistorted normalized image plane
template<typename T, typename T1, typename T2>
inline void undist_img_to_undist_norm(const CameraGeometryBase<T> &camera_geometry, const T1 &pt_undist_img,
                                      T2 &pt_undist_norm) {
    static_assert(is_valid_vector<T1, 2>::value,
                  "Invalid input point type or wrong dimensions. Must have at least 2 dimensions!");
    static_assert(is_valid_vector<T2, 2>::value,
                  "Invalid output point type or wrong dimensions. Must have at least 2 dimensions!");

    using Base         = CameraGeometryBase<T>;
    using Vec2         = typename Base::Vec2;
    using Vec2Map      = Eigen::Map<Vec2>;
    using Vec2MapConst = Eigen::Map<const Vec2>;

    camera_geometry.undist_img_to_undist_norm(Vec2MapConst(get_raw_ptr(pt_undist_img)),
                                              Vec2Map(get_raw_ptr(pt_undist_norm)));
}

/// @brief Maps a point from the undistorted normalized image plane into the distorted image plane
/// @tparam T The floating point type used
/// @tparam T1 The input 2D point type used
/// @tparam T2 The output 2D point type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the point
/// @param[in] pt_undist_norm The point in the undistorted normalized image plane
/// @param[out] pt_dist_img The mapped point in the distorted image plane
template<typename T, typename T1, typename T2>
inline void undist_norm_to_img(const CameraGeometryBase<T> &camera_geometry, const T1 &pt_undist_norm,
                               T2 &pt_dist_img) {
    static_assert(is_valid_vector<T1, 2>::value,
                  "Invalid input point type or wrong dimensions. Must have at least 2 dimensions!");
    static_assert(is_valid_vector<T2, 2>::value,
                  "Invalid output point type or wrong dimensions. Must have at least 2 dimensions!");

    using Base         = CameraGeometryBase<T>;
    using Vec2         = typename Base::Vec2;
    using Vec2Map      = Eigen::Map<Vec2>;
    using Vec2MapConst = Eigen::Map<const Vec2>;

    camera_geometry.undist_norm_to_img(Vec2MapConst(get_raw_ptr(pt_undist_norm)), Vec2Map(get_raw_ptr(pt_dist_img)));
}

/// @brief Maps a point from the distorted image plane into the undistorted normalized image plane
/// @tparam T The floating point type used
/// @tparam T1 The input 2D point type used
/// @tparam T2 The output 2D point type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the point
/// @param[in] pt_dist_img The point in the distorted image plane
/// @param[out] pt_undist_norm The mapped point in the undistorted normalized image plane
template<typename T, typename T1, typename T2>
inline void img_to_undist_norm(const CameraGeometryBase<T> &camera_geometry, const T1 &pt_dist_img,
                               T2 &pt_undist_norm) {
    static_assert(is_valid_vector<T1, 2>::value,
                  "Invalid input point type or wrong dimensions. Must have at least 2 dimensions!");
    static_assert(is_valid_vector<T2, 2>::value,
                  "Invalid output point type or wrong dimensions. Must have at least 2 dimensions!");

    using Base         = CameraGeometryBase<T>;
    using Vec2         = typename Base::Vec2;
    using Vec2Map      = Eigen::Map<Vec2>;
    using Vec2MapConst = Eigen::Map<const Vec2>;

    camera_geometry.img_to_undist_norm(Vec2MapConst(get_raw_ptr(pt_dist_img)), Vec2Map(get_raw_ptr(pt_undist_norm)));
}

/// @brief Maps a point from the undistorted normalized image plane to the distorted normalized image plane
/// @tparam T The floating point type used
/// @tparam T1 The input 2D point type used
/// @tparam T2 The output 2D point type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the point
/// @param[in] pt_undist_norm The point in the undistorted normalized image plane<Paste>
/// @param[out] pt_dist_norm The mapped point in the distorted normalized image plane
template<typename T, typename T1, typename T2>
inline void undist_norm_to_dist_norm(const CameraGeometryBase<T> &camera_geometry, const T1 &pt_undist_norm,
                                     T2 &pt_dist_norm) {
    static_assert(is_valid_vector<T1, 2>::value,
                  "Invalid input point type or wrong dimensions. Must have at least 2 dimensions!");
    static_assert(is_valid_vector<T2, 2>::value,
                  "Invalid output point type or wrong dimensions. Must have at least 2 dimensions!");

    using Base         = CameraGeometryBase<T>;
    using Vec2         = typename Base::Vec2;
    using Vec2Map      = Eigen::Map<Vec2>;
    using Vec2MapConst = Eigen::Map<const Vec2>;

    camera_geometry.undist_norm_to_dist_norm(Vec2MapConst(get_raw_ptr(pt_undist_norm)),
                                             Vec2Map(get_raw_ptr(pt_dist_norm)));
}

/// @brief Maps a vector from the undistorted normalized image plane into the distorted image plane
/// @tparam T The floating point type used
/// @tparam T1 The input 2D point type used
/// @tparam T2 The input 2D vector type used
/// @tparam T3 The output 2D point type used
/// @tparam T4 The output 2D vector type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the vector
/// @param[in] ctr_undist_norm The vector's starting point in the undistorted normalized image plane
/// @param[in] vec_undist_norm The vector in the undistorted normalized image plane (the vector must be normalized)
/// @param[out] ctr_dist_img The vector's starting point in the distorted image plane
/// @param[out] vec_dist_img The mapped vector in the distorted image plane
/// @note The output vector is normalized
template<typename T, typename T1, typename T2, typename T3, typename T4>
inline void vector_undist_norm_to_img(const CameraGeometryBase<T> &camera_geometry, const T1 &ctr_undist_norm,
                                      const T2 &vec_undist_norm, T3 &ctr_dist_img, T4 &vec_dist_img) {
    static_assert(is_valid_vector<T1, 2>::value,
                  "Invalid input point type or wrong dimensions. Must have at least 2 dimensions!");
    static_assert(is_valid_vector<T2, 2>::value,
                  "Invalid input vector type or wrong dimensions. Must have at least 2 dimensions!");
    static_assert(is_valid_vector<T3, 2>::value,
                  "Invalid output point type or wrong dimensions. Must have at least 2 dimensions!");
    static_assert(is_valid_vector<T4, 2>::value,
                  "Invalid output vector type or wrong dimensions. Must have at least 2 dimensions!");

    using Base         = CameraGeometryBase<T>;
    using Vec2         = typename Base::Vec2;
    using Vec2Map      = Eigen::Map<Vec2>;
    using Vec2MapConst = Eigen::Map<const Vec2>;

    camera_geometry.vector_undist_norm_to_img(Vec2MapConst(get_raw_ptr(ctr_undist_norm)),
                                              Vec2MapConst(get_raw_ptr(vec_undist_norm)),
                                              Vec2Map(get_raw_ptr(ctr_dist_img)), Vec2Map(get_raw_ptr(vec_dist_img)));
}

/// @brief Maps a vector from the distorted image plane into the undistorted normalized image plane
/// @tparam T The floating point type used
/// @tparam T1 The input 2D point type used
/// @tparam T2 The input 2D vector type used
/// @tparam T3 The output 2D point type used
/// @tparam T4 The output 2D vector type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the vector
/// @param[in] ctr_dist_img The vector's starting point in the distorted image plane
/// @param[in] vec_dist_img The vector in the distorted image plane (the vector must be normalized)
/// @param[out] ctr_undist_norm The vector's starting point in the undistorted normalized image plane
/// @param[out] vec_undist_norm The vector in the undistorted normalized image plane
/// @note The output vector is normalized
template<typename T, typename T1, typename T2, typename T3, typename T4>
inline void vector_img_to_undist_norm(const CameraGeometryBase<T> &camera_geometry, const T1 &ctr_dist_img,
                                      const T2 &vec_dist_img, T3 &ctr_undist_norm, T4 &vec_undist_norm) {
    static_assert(is_valid_vector<T1, 2>::value,
                  "Invalid input point type or wrong dimensions. Must have at least 2 dimensions!");
    static_assert(is_valid_vector<T2, 2>::value,
                  "Invalid input vector type or wrong dimensions. Must have at least 2 dimensions!");
    static_assert(is_valid_vector<T3, 2>::value,
                  "Invalid output point type or wrong dimensions. Must have at least 2 dimensions!");
    static_assert(is_valid_vector<T4, 2>::value,
                  "Invalid output vector type or wrong dimensions. Must have at least 2 dimensions!");

    using Base         = CameraGeometryBase<T>;
    using Vec2         = typename Base::Vec2;
    using Vec2Map      = Eigen::Map<Vec2>;
    using Vec2MapConst = Eigen::Map<const Vec2>;

    camera_geometry.vector_img_to_undist_norm(
        Vec2MapConst(get_raw_ptr(ctr_dist_img)), Vec2MapConst(get_raw_ptr(vec_dist_img)),
        Vec2Map(get_raw_ptr(ctr_undist_norm)), Vec2Map(get_raw_ptr(vec_undist_norm)));
}

/// @brief Computes the distortion function's jacobian
/// @tparam T The floating point type used
/// @tparam T1 The input 2D point type used
/// @tparam T2 The output 2D point type used
/// @tparam T3 The output 2x2 matrix type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the point
/// @param[in] pt_undist_norm The point in the undistorted normalized image plane at which the jacobian is computed
/// @param[out] pt_dist_img The point in the distorted image plane
/// @param[out] J The computed jacobian
template<typename T, typename T1, typename T2, typename T3>
inline void get_undist_norm_to_img_jacobian(const CameraGeometryBase<T> &camera_geometry, const T1 &pt_undist_norm,
                                            T2 &pt_dist_img, T3 &J) {
    static_assert(is_valid_vector<T1, 2>::value,
                  "Invalid input point type or wrong dimensions. Must have at least 2 dimensions!");
    static_assert(is_valid_vector<T2, 2>::value,
                  "Invalid input vector type or wrong dimensions. Must have at least 2 dimensions!");
    static_assert(is_valid_matrix<T3, 2, 2>::value,
                  "Invalid output matrix type or wrong dimensions. Must be at least of dimensions 2x2!");

    using Base         = CameraGeometryBase<T>;
    using Vec2         = typename Base::Vec2;
    using Vec2Map      = Eigen::Map<Vec2>;
    using Vec2MapConst = Eigen::Map<const Vec2>;
    using Mat2    = std::conditional_t<detail::is_row_major<T3>::value, typename Base::Mat2RM, typename Base::Mat2CM>;
    using Mat2Map = Eigen::Map<Mat2>;

    camera_geometry.get_undist_norm_to_img_jacobian(Vec2MapConst(get_raw_ptr(pt_undist_norm)),
                                                    Vec2Map(get_raw_ptr(pt_dist_img)), Mat2Map(get_mat_raw_ptr(J)));
}

/// @brief Computes the undistortion function's jacobian
/// @tparam T The floating point type used
/// @tparam T1 The input 2D point type used
/// @tparam T2 The output 2D point type used
/// @tparam T3 The output 2x2 matrix type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the point
/// @param[in] pt_dist_img The point in the distorted image plane at which the jacobian is computed
/// @param[out] pt_undist_norm The point in the undistorted normalized image plane
/// @param[out] J The computed jacobian
template<typename T, typename T1, typename T2, typename T3>
inline void get_img_to_undist_norm_jacobian(const CameraGeometryBase<T> &camera_geometry, const T1 &pt_dist_img,
                                            T2 &pt_undist_norm, T3 &J) {
    static_assert(is_valid_vector<T1, 2>::value,
                  "Invalid input point type or wrong dimensions. Must have at least 2 dimensions!");
    static_assert(is_valid_vector<T2, 2>::value,
                  "Invalid input vector type or wrong dimensions. Must have at least 2 dimensions!");
    static_assert(is_valid_matrix<T3, 2, 2>::value,
                  "Invalid output matrix type or wrong dimensions. Must be at least of dimensions 2x2!");

    using Base         = CameraGeometryBase<T>;
    using Vec2         = typename Base::Vec2;
    using Vec2Map      = Eigen::Map<Vec2>;
    using Vec2MapConst = Eigen::Map<const Vec2>;
    using Mat2    = std::conditional_t<detail::is_row_major<T3>::value, typename Base::Mat2RM, typename Base::Mat2CM>;
    using Mat2Map = Eigen::Map<Mat2>;

    camera_geometry.get_img_to_undist_norm_jacobian(Vec2MapConst(get_raw_ptr(pt_dist_img)),
                                                    Vec2Map(get_raw_ptr(pt_undist_norm)), Mat2Map(get_mat_raw_ptr(J)));
}

/// @brief Computes the distortion maps which are LUT used to distort coordinates
///
/// The LUT are NxM matrices where NxM is the sensor's resolution. The LUT must be allocated before calling this
/// function.
///
/// @warning Such maps are used with cv::remap to undistort an image.
/// @tparam T The floating point type used
/// @tparam T1 The output matrix type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the point
/// @param[out] mapx LUT giving the distorted x coordinate of the input undistorted point (integer coordinates)
/// @param[out] mapy LUT giving the distorted y coordinate of the input undistorted point (integer coordinates)
template<typename T, typename T1>
inline void get_distortion_maps(const CameraGeometryBase<T> &camera_geometry, T1 &mapx, T1 &mapy) {
    static_assert(is_valid_dynamic_matrix<T1>::value, "Unsupported matrix type or fixed size matrix!");

    using Base = CameraGeometryBase<T>;
    using Vec2 = typename Base::Vec2;

    const auto &img_size = camera_geometry.get_image_size();

    Vec2 v1, v2;

    for (int row = 0; row < img_size(1); ++row) {
        for (int col = 0; col < img_size(0); ++col) {
            get_x_element(v1) = T(col);
            get_y_element(v1) = T(row);

            camera_geometry.undist_img_to_undist_norm(v1, v2);
            camera_geometry.undist_norm_to_img(v2, v1);

            get_mat_element(mapx, row, col) = get_x_element(v1);
            get_mat_element(mapy, row, col) = get_y_element(v1);
        }
    }
}

/// @brief Computes the undistortion maps which are LUT used to undistort coordinates
///
/// The LUT are NxM matrices where NxM is the sensor's resolution. The LUT must be allocated before calling this
/// function.
///
/// @warning Such maps are used with cv::remap to distort an image.
/// @tparam T The floating point type used
/// @tparam T1 The output matrix type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the point
/// @param[out] mapx LUT giving the undistorted x coordinate of the input distorted point (integer coordinates)
/// @param[out] mapy LUT giving the undistorted y coordinate of the input distorted point (integer coordinates)
template<typename T, typename T1>
inline void get_undistortion_maps(const CameraGeometryBase<T> &camera_geometry, T1 &mapx, T1 &mapy) {
    static_assert(is_valid_dynamic_matrix<T1>::value, "Unsupported matrix type or fixed size matrix!");

    using Base = CameraGeometryBase<T>;
    using Vec2 = typename Base::Vec2;

    const auto &img_size = camera_geometry.get_image_size();

    Vec2 v1, v2;

    for (int row = 0; row < img_size(1); ++row) {
        for (int col = 0; col < img_size(0); ++col) {
            get_x_element(v1) = T(col);
            get_y_element(v1) = T(row);

            camera_geometry.img_to_undist_norm(v1, v2);
            camera_geometry.undist_norm_to_undist_img(v2, v1);

            get_mat_element(mapx, row, col) = get_x_element(v1);
            get_mat_element(mapy, row, col) = get_y_element(v1);
        }
    }
}

/// @brief Computes the LUT used to apply perspective and distort coordinates. This maps the undistorted fronto-parallel
/// view back to the distorted perspective view
///
/// The LUT are NxM matrices where NxM is the sensor's resolution. The LUT must be allocated before calling this
/// function.
///
/// @warning Such maps are used with cv::remap to both undistort and unproject an image to the fronto-parallel view.
/// Since cv::remap requires the reverse transformation, the homography @p H has to map back from the fronto-parallel to
/// the perspective view
/// @tparam T The floating point type used
/// @tparam T1 The input 3x3 matrix type used
/// @tparam T2 The output matrix type used
/// @param[in] camera_geometry The CameraGeometryBase instance used to map the point
/// @param[in] H 3x3 Homography matrix mapping the undistorted fronto-parallel view back to the undistorted perspective
/// view
/// @param[out] mapx LUT giving the distorted x coordinate of the input undistorted point (integer coordinates)
/// @param[out] mapy LUT giving the distorted y coordinate of the input undistorted point (integer coordinates)
template<typename T, typename T1, typename T2>
inline void get_homography_and_distortion_maps(const CameraGeometryBase<T> &camera_geometry, const T1 &H, T2 &mapx,
                                               T2 &mapy) {
    static_assert(is_valid_matrix<T1, 3, 3>::value,
                  "Invalid output matrix type or wrong dimensions. Must be at least of dimensions 3x3!");
    static_assert(is_valid_dynamic_matrix<T2>::value, "Unsupported matrix type or fixed size matrix!");

    using Base = CameraGeometryBase<T>;
    using Vec2 = typename Base::Vec2;

    const auto &img_size = camera_geometry.get_image_size();

    Vec2 v1, v2;

    for (int row = 0; row < img_size(1); ++row) {
        for (int col = 0; col < img_size(0); ++col) {
            const T x = H(0, 0) * T(col) + H(0, 1) * T(row) + H(0, 2);
            const T y = H(1, 0) * T(col) + H(1, 1) * T(row) + H(1, 2);
            const T z = H(2, 0) * T(col) + H(2, 1) * T(row) + H(2, 2);

            get_x_element(v1) = x / z;
            get_y_element(v1) = y / z;

            camera_geometry.undist_img_to_undist_norm(v1, v2);
            camera_geometry.undist_norm_to_img(v2, v1);

            get_mat_element(mapx, row, col) = get_x_element(v1);
            get_mat_element(mapy, row, col) = get_y_element(v1);
        }
    }
}

/// @brief For a dual camera setup, computes the maps used to map an imput image from one sensor to another sensor
/// given a homography between the two.
///
/// The maps are NxM matrices where NxM is the destination sensor's resolution. The maps must be allocated before
/// calling this function.
///
/// @warning Such maps are used with cv::remap to change the sensor on which image is projected.
/// Since cv::remap requires the reverse transformation, the homography @p H has to map back from the dst sensor to the
/// src sensor
/// @tparam T The floating point type used
/// @tparam T1 The input 3x3 matrix type used
/// @tparam T2 The output matrix type used
/// @param[in] src_cam_geometry The CameraGeometryBase instance of the source camera
/// @param[in] dst_cam_geometry The CameraGeometryBase instance of the destination camera
/// @param[in] H_src_dst 3x3 Homography matrix mapping the undistorted normalized plan of the destination sensor
/// back to the undistorted normalized plan of the source sensor : v_src = H_src_dst*v_dst
/// @param[out] mapx Map giving the x coordinate of the src sensor point associated with the coordinates of the
/// destination sensor (float coordinates)
/// @param[out] mapy Map giving the y coordinate of the src sensor point associated with the coordinates of the
/// destination sensor (float coordinates)
template<typename T, typename T1, typename T2>
inline void compute_camera_transfer_maps(const CameraGeometryBase<T> &src_cam_geometry,
                                         const CameraGeometryBase<T> &dst_cam_geometry, const T1 &H_src_dst, T2 &mapx,
                                         T2 &mapy) {
    static_assert(is_valid_matrix<T1, 3, 3>::value,
                  "Invalid output matrix type or wrong dimensions. Must be at least of dimensions 3x3!");
    static_assert(is_valid_dynamic_matrix<T2>::value, "Unsupported matrix type or fixed size matrix!");

    using Base = CameraGeometryBase<T>;
    using Vec2 = typename Base::Vec2;

    const auto &dst_size = dst_cam_geometry.get_image_size();
    const auto &src_size = src_cam_geometry.get_image_size();

    Vec2 v1, v2;

    for (int row = 0; row < dst_size(1); ++row) {
        for (int col = 0; col < dst_size(0); ++col) {
            get_x_element(v1) = T(col);
            get_y_element(v1) = T(row);

            dst_cam_geometry.img_to_undist_norm(v1, v2);

            const T x = H_src_dst(0, 0) * v2(0) + H_src_dst(0, 1) * v2(1) + H_src_dst(0, 2);
            const T y = H_src_dst(1, 0) * v2(0) + H_src_dst(1, 1) * v2(1) + H_src_dst(1, 2);
            const T z = H_src_dst(2, 0) * v2(0) + H_src_dst(2, 1) * v2(1) + H_src_dst(2, 2);

            get_x_element(v1) = x / z;
            get_y_element(v1) = y / z;

            src_cam_geometry.undist_norm_to_img(v1, v2);

            get_mat_element(mapx, row, col) = get_x_element(v2);
            get_mat_element(mapy, row, col) = get_y_element(v2);
        }
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CV_CAMERA_GEOMETRY_HELPERS_H
