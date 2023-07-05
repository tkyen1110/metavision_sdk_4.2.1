/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_CAMERA_GEOMETRY_BASE_H
#define METAVISION_SDK_CV_CAMERA_GEOMETRY_BASE_H

#include <type_traits>
#include <memory>
#include <Eigen/Core>

namespace Metavision {

/// @brief Base class for camera geometries
///
/// A camera geometry is a mathematical model allowing to map points from world to image plane and vice versa.
/// As this class uses virtual methods, it was not possible to make them templated. Therefore, eigen has been used for
/// the vector and matrices types because it allows to map other types without copies (i.e. by using a combination of
/// Eigen::Map and Eigen::Ref).
///
/// @tparam T Either float or double
template<typename T>
class CameraGeometryBase {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static_assert(std::is_floating_point<T>::value, "Float type required");
    using Vec2         = Eigen::Matrix<T, 2, 1>;
    using Vec3         = Eigen::Matrix<T, 3, 1>;
    using Vec2Ref      = Eigen::Ref<Vec2, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
    using Vec2RefConst = const Eigen::Ref<const Vec2, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
    using Vec3RefConst = const Eigen::Ref<const Vec3, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
    using Vec2i        = Eigen::Vector2i;
    using Mat2RM       = Eigen::Matrix<T, 2, 2, Eigen::RowMajor>;
    using Mat2CM       = Eigen::Matrix<T, 2, 2, Eigen::ColMajor>;
    using Mat3RM       = Eigen::Matrix<T, 3, 3, Eigen::RowMajor>;
    using Mat3CM       = Eigen::Matrix<T, 3, 3, Eigen::ColMajor>;
    using Mat2RMRef    = Eigen::Ref<Mat2RM>;
    using Mat2CMRef    = Eigen::Ref<Mat2CM>;
    using Mat3RMRef    = Eigen::Ref<Mat3RM>;
    using Mat3CMRef    = Eigen::Ref<Mat3CM>;

    /// @brief Default destructor
    virtual ~CameraGeometryBase() = default;

    /// @brief Creates a deep copy of this instance
    /// @return The new instance
    virtual std::unique_ptr<CameraGeometryBase<T>> clone() const = 0;

    /// @brief Gets the sensor's size
    virtual const Vec2i &get_image_size() const = 0;

    /// @brief Gets the distance between the camera's optical center and the undistorted image plane
    virtual T get_distance_to_image_plane() const = 0;

    /// @brief Gets the transform that maps a point from the undistorted normalized image plane (i.e. Z = 1) into the
    /// undistorted image plane (row major mode matrix)
    /// @param[out] m The transform
    virtual void get_undist_norm_to_undist_img_transform(Mat3RMRef m) const = 0;

    /// @brief Gets the transform that maps a point from the undistorted normalized image plane (i.e. Z = 1) into the
    /// undistorted image plane (col major mode matrix)
    /// @param[out] m The transform
    virtual void get_undist_norm_to_undist_img_transform(Mat3CMRef m) const = 0;

    /// @brief Maps a point from the camera's coordinates system into the distorted image plane
    /// @param[in] pt_c The 3D point in the camera's coordinates system
    /// @param[out] pt_dist_img The mapped point in the distorted image plane
    virtual void camera_to_img(Vec3RefConst pt_c, Vec2Ref pt_dist_img) const = 0;

    /// @brief Maps a point from the camera's coordinates system into the undistorted image plane
    /// @param[in] pt_c The 3D point in the camera's coordinates system
    /// @param[out] pt_undist_img The mapped point in the undistorted image plane
    virtual void camera_to_undist_img(Vec3RefConst pt_c, Vec2Ref pt_undist_img) const = 0;

    /// @brief Maps a point from the undistorted normalized image plane into the normalized image plane
    /// @param[in] pt_undist_norm The point in the undistorted normalized image plane
    /// @param[out] pt_undist_img The mapped point in the undistorted image plane
    virtual void undist_norm_to_undist_img(Vec2RefConst pt_undist_norm, Vec2Ref pt_undist_img) const = 0;

    /// @brief Maps a point from the undistorted image plane into the undistorted normalized image plane
    /// @param[in] pt_undist_img The point in the undistorted image plane
    /// @param[out] pt_undist_norm The mapped point in the undistorted normalized image plane
    virtual void undist_img_to_undist_norm(Vec2RefConst pt_undist_img, Vec2Ref pt_undist_norm) const = 0;

    /// @brief Maps a point from the undistorted normalized image plane into the distorted image plane
    /// @param[in] pt_undist_norm The point in the undistorted normalized image plane
    /// @param[out] pt_dist_img The mapped point in the distorted image plane
    virtual void undist_norm_to_img(Vec2RefConst pt_undist_norm, Vec2Ref pt_dist_img) const = 0;

    /// @brief Maps a point from the distorted image plane into the undistorted normalized image plane
    /// @param[in] pt_dist_img The point in the distorted image plane
    /// @param[out] pt_undist_norm The mapped point in the undistorted normalized image plane
    virtual void img_to_undist_norm(Vec2RefConst pt_dist_img, Vec2Ref pt_undist_norm) const = 0;

    /// @brief Maps a point from the undistorted normalized image plane into the distorted normalized image plane
    /// @param[in] pt_undist_norm The mapped point in the undistorted normalized image plane
    /// @param[out] pt_dist_norm The mapped point in the distorted normalized image plane
    virtual void undist_norm_to_dist_norm(Vec2RefConst pt_undist_norm, Vec2Ref pt_dist_norm) const = 0;

    /// @brief Maps a vector from the undistorted normalized image plane into the distorted image plane
    /// @param[in] ctr_undist_norm The vector's starting point in the undistorted normalized image plane
    /// @param[in] vec_undist_norm The vector in the undistorted normalized image plane (the vector must be normalized)
    /// @param[out] ctr_dist_img The vector's starting point in the distorted image plane
    /// @param[out] vec_dist_img The mapped vector in the distorted image plane
    /// @note The output vector is normalized
    virtual void vector_undist_norm_to_img(Vec2RefConst ctr_undist_norm, Vec2RefConst vec_undist_norm,
                                           Vec2Ref ctr_dist_img, Vec2Ref vec_dist_img) const = 0;

    /// @brief Maps a vector from the distorted image plane into the undistorted normalized image plane
    /// @param[in] ctr_dist_img The vector's starting point in the distorted image plane
    /// @param[in] vec_dist_img The vector in the distorted image plane (the vector must be normalized)
    /// @param[out] ctr_undist_norm The vector's starting point in the undistorted normalized image plane
    /// @param[out] vec_undist_norm The vector in the undistorted normalized image plane
    /// @note The output vector is normalized
    virtual void vector_img_to_undist_norm(Vec2RefConst ctr_dist_img, Vec2RefConst vec_dist_img,
                                           Vec2Ref ctr_undist_norm, Vec2Ref vec_undist_norm) const = 0;

    /// @brief Computes the distortion function's jacobian (Row major mode matrix)
    /// @param[in] pt_undist_norm The point in the undistorted normalized image plane at which the jacobian is computed
    /// @param[out] pt_dist_img The point in the distorted image plane
    /// @param[out] J The computed jacobian
    virtual void get_undist_norm_to_img_jacobian(Vec2RefConst pt_undist_norm, Vec2Ref pt_dist_img,
                                                 Mat2RMRef J) const = 0;

    /// @brief Computes the distortion function's jacobian (Col major mode matrix)
    /// @param[in] pt_undist_norm The point in the undistorted normalized image plane at which the jacobian is computed
    /// @param[out] pt_dist_img The point in the distorted image plane
    /// @param[out] J The computed jacobian
    virtual void get_undist_norm_to_img_jacobian(Vec2RefConst pt_undist_norm, Vec2Ref pt_dist_img,
                                                 Mat2CMRef J) const = 0;

    /// @brief Computes the undistortion function's jacobian (Row major mode matrix)
    /// @param[in] pt_dist_img The point in the distorted image plane at which the jacobian is computed
    /// @param[out] pt_undist_norm The point in the undistorted normalized image plane
    /// @param[out] J The computed jacobian
    virtual void get_img_to_undist_norm_jacobian(Vec2RefConst pt_dist_img, Vec2Ref pt_undist_norm,
                                                 Mat2RMRef J) const = 0;

    /// @brief Computes the undistortion function's jacobian (Col major mode matrix)
    /// @param[in] pt_dist_img The point in the distorted image plane at which the jacobian is computed
    /// @param[out] pt_undist_norm The point in the undistorted normalized image plane
    /// @param[out] J The computed jacobian
    virtual void get_img_to_undist_norm_jacobian(Vec2RefConst pt_dist_img, Vec2Ref pt_undist_norm,
                                                 Mat2CMRef J) const = 0;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_CAMERA_GEOMETRY_BASE_H
