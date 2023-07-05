/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_PINHOLE_CAMERA_MODEL_H
#define METAVISION_SDK_CV_PINHOLE_CAMERA_MODEL_H

#include <Eigen/Core>

namespace Metavision {

/// @brief Class implementing the Pinhole camera model with radial and tangential distortion
/// @tparam FloatType Either float or double
template<typename FloatType>
class PinholeCameraModel {
public:
    static_assert(std::is_floating_point<FloatType>::value, "Float type required");

    using underlying_type = FloatType;
    using Vec2i           = Eigen::Vector2i;
    using Mat3            = Eigen::Matrix<FloatType, 3, 3>;
    using Vec5            = Eigen::Matrix<FloatType, 5, 1>;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// @brief Builds a new @ref PinholeCameraModel
    /// @param[in] width Sensor's width
    /// @param[in] height Sensor's height
    /// @param[in] K Camera's intrinsics 3x3 matrix
    /// @param[in] D Camera's distortion coefficients
    PinholeCameraModel(int width, int height, const std::vector<FloatType> &K, const std::vector<FloatType> &D);

    /// @brief Gets the sensor's size
    const Vec2i &get_image_size() const;

    /// @brief Gets the camera's intrinsics
    const Mat3 &K() const;

    /// @brief Gets the camera's inverse intrinsics
    const Mat3 &Kinv() const;

    /// @brief Gets the camera's distortion coefficients
    const Vec5 &D() const;

    /// @brief Gets the distance between the camera's optical center and the undistorted image plane
    FloatType get_distance_to_image_plane() const;

    /// @brief Gets the transform that maps a point from the undistorted normalized image plane (i.e. Z = 1) into the
    /// undistorted image plane
    /// @tparam M The 3x3 matrix's type used
    /// @param[out] m The transform
    template<typename M>
    void get_undist_norm_to_undist_img_transform(M &m) const;

    /// @brief Maps a point from the camera's coordinates system into the distorted image plane
    /// @tparam V1 The 3D point's type used
    /// @tparam V2 The 2D point's type used
    /// @param[in] pt_c The 3D point in the camera's coordinates system
    /// @param[out] pt_dist_img The mapped point in the distorted image plane
    template<typename V1, typename V2>
    inline void camera_to_img(const V1 &pt_c, V2 &pt_dist_img) const;

    /// @brief Maps a point from the undistorted normalized image plane into the distorted image plane
    /// @tparam V1 The input 2D point's type used
    /// @tparam V2 The output 2D point's type used
    /// @param[in] pt_undist_norm The point in the undistorted normalized image plane
    /// @param[out] pt_dist_img The mapped point in the distorted image plane
    template<typename V1, typename V2>
    inline void undist_norm_to_img(const V1 &pt_undist_norm, V2 &pt_dist_img) const;

    /// @brief Maps a point from the camera's coordinates system into the undistorted image plane
    /// @tparam V1 The 3D point's type used
    /// @tparam V2 The 2D point's type used
    /// @param[in] pt_c The 3D point in the camera's coordinates system
    /// @param[out] pt_undist_img The mapped point in the undistorted image plane
    template<typename V1, typename V2>
    inline void camera_to_undist_img(const V1 &pt_c, V2 &pt_undist_img) const;

    /// @brief Maps a point from the undistorted normalized image plane into the undistorted image plane
    /// @tparam V1 The input 2D point's type used
    /// @tparam V2 The output 2D point's type used
    /// @param[in] pt_undist_norm The point in the undistorted normalized image plane
    /// @param[out] pt_undist_img The mapped point in the undistorted image plane
    template<typename V1, typename V2>
    inline void undist_norm_to_undist_img(const V1 &pt_undist_norm, V2 &pt_undist_img) const;

    /// @brief Maps a point from the undistorted image plane into the undistorted normalized image plane
    /// @tparam V1 The input 2D point's type used
    /// @tparam V2 The output 2D point's type used
    /// @param[in] pt_undist_img The point in the undistorted image plane
    /// @param[out] pt_undist_norm The mapped point in the undistorted normalized image plane
    template<typename V1, typename V2>
    inline void undist_img_to_undist_norm(const V1 &pt_undist_img, V2 &pt_undist_norm) const;

    /// @brief Maps a point from the undistorted normalized image plane into the distorted normalized image plane
    /// @tparam V1 The input 2D point's type used
    /// @tparam V2 The output 2D point's type used
    /// @param[in] pt_undist_norm The mapped point in the undistorted normalized image plane
    /// @param[out] pt_dist_norm The mapped point in the distorted normalized image plane
    template<typename V1, typename V2>
    inline void undist_norm_to_dist_norm(const V1 &pt_undist_norm, V2 &pt_dist_norm) const;

    /// @brief Maps a point from the distorted image plane into the undistorted normalized image plane
    /// @tparam V1 The input 2D point's type used
    /// @tparam V2 The output 2D point's type used
    /// @param[in] pt_dist_img The point in the distorted image plane
    /// @param[out] pt_undist_norm The mapped point in the undistorted normalized image plane
    template<typename V1, typename V2>
    inline void img_to_undist_norm(const V1 &pt_dist_img, V2 &pt_undist_norm) const;

    /// @brief Computes the distortion function's jacobian
    /// @tparam V1 The input 2D point's type used
    /// @tparam V2 The output 2D point's type used
    /// @tparam V3 The output 2x2 matrix's type used
    /// @param[in] pt_undist_norm The point in the undistorted normalized image plane at which the jacobian is computed
    /// @param[out] pt_dist_img The point in the distorted image plane
    /// @param[out] J The computed jacobian
    template<typename V1, typename V2, typename M>
    inline void get_undist_norm_to_img_jacobian(const V1 &pt_undist_norm, V2 &pt_dist_img, M &J) const;

    /// @brief Computes the undistortion function's jacobian
    /// @tparam V1 The input 2D point's type used
    /// @tparam V2 The output 2D point's type used
    /// @tparam V3 The output 2x2 matrix's type used
    /// @param[in] pt_dist_img The point in the distorted image plane at which the jacobian is computed
    /// @param[out] pt_undist_norm The point in the undistorted normalized image plane
    /// @param[out] J The computed jacobian
    template<typename V1, typename V2, typename M>
    inline void get_img_to_undist_norm_jacobian(const V1 &pt_dist_img, V2 &pt_undist_norm, M &J) const;

private:
    template<typename V1, typename V2>
    inline void camera_to_undist_norm(const V1 &pt_c, V2 &pt_undist_norm) const;

    template<typename V1, typename V2>
    inline void norm_to_img(const V1 &pt_norm, V2 &pt_img) const;

    template<typename V1, typename V2>
    inline void img_to_norm(const V1 &pt_img, V2 &pt_norm) const;

    template<typename V1, typename V2>
    inline void dist_norm_to_undist_norm(const V1 &pt_dist_norm, V2 &pt_undist_norm) const;

    Vec2i img_size_; ///< Sensor's size
    Mat3 K_;         ///< Camera's intrinsics
    Mat3 Kinv_;      ///< Camera's inverse intrinsics
    Vec5 D_;         ///< Camera's distortion coefficients
};

} // namespace Metavision

#include "metavision/sdk/cv/utils/detail/pinhole_camera_model_impl.h"

#endif // METAVISION_SDK_CV_PINHOLE_CAMERA_MODEL_H
