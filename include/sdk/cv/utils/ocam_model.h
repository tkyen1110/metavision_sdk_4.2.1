/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_OCAM_MODEL_H
#define METAVISION_SDK_CV_OCAM_MODEL_H

#include <type_traits>
#include <Eigen/Core>

namespace Metavision {

/// @brief Class implementing the Scaramuzza's fisheye camera model
///
/// ("A Toolbox for Easily Calibrating Omnidirectional Cameras").
/// The projection and distortion are done simultaneously.
///
/// @tparam FloatType Either float or double.
template<typename FloatType>
class OcamModel {
public:
    static_assert(std::is_floating_point<FloatType>::value, "Float type required");

    using underlying_type = FloatType;
    using Vec2            = Eigen::Matrix<FloatType, 2, 1>;
    using Vec2i           = Eigen::Vector2i;
    using Mat2CM          = Eigen::Matrix<FloatType, 2, 2>;
    using VecX            = Eigen::Matrix<FloatType, Eigen::Dynamic, 1>;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// @brief Builds a new @ref OcamModel
    ///
    /// The intrinsics must correspond to a camera having:
    ///     - its X axis pointing to the right
    ///     - its Y axis pointing down, and
    ///     - its Z axis pointing toward
    ///
    /// @param[in] img_size Sensor's size
    /// @param[in] poly Polynomial used to undistort coordinates
    /// @param[in] inv_poly Polynomial used to distort coordinates
    /// @param[in] center Projection of the camera's optical center in the image plane
    /// @param[in] affine_transform Transform that maps from the ideal sensor plane to the image plane
    /// @param[in] zoom_factor Scale factor indicating how the image appears once undistorted. The bigger is the
    /// factor, the smaller is the undistorted image (i.e. the more black areas in the image).
    OcamModel(const Vec2i &img_size, const VecX &poly, const VecX &inv_poly, const Vec2 &center,
              const Mat2CM &affine_transform, FloatType zoom_factor = FloatType(1));

    /// @brief Gets the sensor's size
    const Vec2i &get_image_size() const;

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
    inline void camera_to_img_internal(const V1 &pt_undist_norm, const FloatType &z, V2 &pt_dist_img) const;

    VecX p_;            ///< Polynomial's coefficient used for undistortion
    VecX i_p_;          ///< Polynomial's coefficient used for distortion (inverse polynomial)
    Mat2CM A_;          ///< Affine transform that maps from the ideal sensor plane to the image plane
    Mat2CM i_A_;        ///< Inverse affine transform
    Vec2 center_;       ///< Optical center's projection on the distorted image plane
    Vec2 ideal_center_; ///< Optical center's projection on the undistorted image plane
    Vec2i img_size_;    ///< Image's size
    FloatType scale_;   ///< Zoom factor, distance of the undistorted image to the camera
};

} // namespace Metavision

#include "metavision/sdk/cv/utils/detail/ocam_model_impl.h"

#endif // METAVISION_SDK_CV_OCAM_MODEL_H
