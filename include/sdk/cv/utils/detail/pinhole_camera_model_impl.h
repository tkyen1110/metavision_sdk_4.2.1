/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_DETAIL_PINHOLE_CAMERA_MODEL_IMPL_H
#define METAVISION_SDK_CV_DETAIL_PINHOLE_CAMERA_MODEL_IMPL_H

#include <Eigen/LU>

#include "metavision/sdk/cv/utils/detail/mat_traits.h"

namespace Metavision {

template<typename FloatType>
PinholeCameraModel<FloatType>::PinholeCameraModel(int width, int height, const std::vector<FloatType> &K,
                                                  const std::vector<FloatType> &D) {
    assert(K.size() == 9);
    assert(D.size() == 4 || D.size() == 5);

    img_size_ << width, height;

    K_    = Eigen::Map<const Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor>>(K.data());
    Kinv_ = K_.inverse();

    D_.setZero();
    std::copy(D.cbegin(), D.cend(), D_.data());
}

template<typename FloatType>
const Eigen::Vector2i &PinholeCameraModel<FloatType>::get_image_size() const {
    return img_size_;
}

template<typename FloatType>
FloatType PinholeCameraModel<FloatType>::get_distance_to_image_plane() const {
    return K_(0, 0);
}

template<typename FloatType>
template<typename M>
void PinholeCameraModel<FloatType>::get_undist_norm_to_undist_img_transform(M &m) const {
    static_assert(mat_traits<M>::dimX == 3 && mat_traits<M>::dimY == 3, "Output matrix must be of dimensions 3x3");

    m = K_;
}

template<typename FloatType>
const Eigen::Matrix<FloatType, 3, 3> &PinholeCameraModel<FloatType>::K() const {
    return K_;
}

template<typename FloatType>
const Eigen::Matrix<FloatType, 3, 3> &PinholeCameraModel<FloatType>::Kinv() const {
    return Kinv_;
}

template<typename FloatType>
const Eigen::Matrix<FloatType, 5, 1> &PinholeCameraModel<FloatType>::D() const {
    return D_;
}

template<typename FloatType>
template<typename V1, typename V2>
inline void PinholeCameraModel<FloatType>::camera_to_img(const V1 &pt_c, V2 &pt_dist_img) const {
    static_assert(vec_traits<V1>::dim == 3, "Input Vector must have 3 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");

    camera_to_undist_norm(pt_c, pt_dist_img);
    undist_norm_to_img(pt_dist_img, pt_dist_img);
}

template<typename FloatType>
template<typename V1, typename V2>
inline void PinholeCameraModel<FloatType>::undist_norm_to_img(const V1 &pt_undist_norm, V2 &pt_dist_img) const {
    static_assert(vec_traits<V1>::dim == 2, "Input Vector must have 2 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");

    undist_norm_to_dist_norm(pt_undist_norm, pt_dist_img);
    norm_to_img(pt_dist_img, pt_dist_img);
}

template<typename FloatType>
template<typename V1, typename V2>
inline void PinholeCameraModel<FloatType>::camera_to_undist_img(const V1 &pt_c, V2 &pt_undist_img) const {
    static_assert(vec_traits<V1>::dim == 3, "Input Vector must have 3 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");

    camera_to_undist_norm(pt_c, pt_undist_img);
    norm_to_img(pt_undist_img, pt_undist_img);
}

template<typename FloatType>
template<typename V1, typename V2>
inline void PinholeCameraModel<FloatType>::undist_norm_to_undist_img(const V1 &pt_undist_norm,
                                                                     V2 &pt_undist_img) const {
    static_assert(vec_traits<V1>::dim == 2, "Input Vector must have 2 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");

    norm_to_img(pt_undist_norm, pt_undist_img);
}

template<typename FloatType>
template<typename V1, typename V2>
inline void PinholeCameraModel<FloatType>::undist_img_to_undist_norm(const V1 &pt_undist_img,
                                                                     V2 &pt_undist_norm) const {
    static_assert(vec_traits<V1>::dim == 2, "Input Vector must have 2 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");

    img_to_norm(pt_undist_img, pt_undist_norm);
}

template<typename FloatType>
template<typename V1, typename V2>
inline void PinholeCameraModel<FloatType>::img_to_undist_norm(const V1 &pt_dist_img, V2 &pt_undist_norm) const {
    static_assert(vec_traits<V1>::dim == 2, "Input Vector must have 2 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");

    img_to_norm(pt_dist_img, pt_undist_norm);
    dist_norm_to_undist_norm(pt_undist_norm, pt_undist_norm);
}

template<typename FloatType>
template<typename V1, typename V2, typename M>
inline void PinholeCameraModel<FloatType>::get_undist_norm_to_img_jacobian(const V1 &pt_undist_norm, V2 &pt_dist_img,
                                                                           M &J) const {
    static_assert(vec_traits<V1>::dim == 2, "Input Vector must have 2 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");
    static_assert(mat_traits<M>::dimX == 2 && mat_traits<M>::dimY == 2, "Output matrix must be of dimensions 2x2");

    undist_norm_to_img(pt_undist_norm, pt_dist_img);

    Eigen::Matrix<FloatType, 2, 2> Jdist;

    const FloatType x  = pt_undist_norm(0);
    const FloatType x2 = x * x;
    const FloatType y  = pt_undist_norm(1);
    const FloatType y2 = y * y;
    const FloatType xy = x * y, r2 = x2 + y2, r4 = r2 * r2, r6 = r4 * r2;
    const FloatType c_raddist       = 1 + D_(0) * r2 + D_(1) * r4 + D_(4) * r6;
    const FloatType c_raddist_deriv = D_(0) + 2 * D_(1) * r2 + 3 * D_(4) * r4;

    Jdist(0, 0) = c_raddist + 2 * x2 * c_raddist_deriv + 2 * D_(2) * y + 6 * D_(3) * x;
    Jdist(0, 1) = 2 * xy * c_raddist_deriv + 2 * D_(2) * x + 2 * D_(3) * y;
    Jdist(1, 0) = Jdist(0, 1);
    Jdist(1, 1) = c_raddist + 2 * y2 * c_raddist_deriv + 6 * D_(2) * y + 2 * D_(3) * x;

    J = K_.template block<2, 2>(0, 0) * Jdist;
}

template<typename FloatType>
template<typename V1, typename V2, typename M>
inline void PinholeCameraModel<FloatType>::get_img_to_undist_norm_jacobian(const V1 &pt_dist_img, V2 &pt_undist_norm,
                                                                           M &J) const {
    static_assert(vec_traits<V1>::dim == 2, "Input Vector must have 2 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");
    static_assert(mat_traits<M>::dimX == 2 && mat_traits<M>::dimY == 2, "Output matrix must be of dimensions 2x2");

    using T = FloatType;

    const T x_dist_norm = Kinv_(0, 0) * pt_dist_img(0) + Kinv_(0, 2);
    const T y_dist_norm = Kinv_(1, 1) * pt_dist_img(1) + Kinv_(1, 2);

    const T &k0        = D_(0);
    const T &k1        = D_(1);
    const T &k2        = D_(2);
    const T &k3        = D_(3);
    const T &k4        = D_(4);
    const T k4_times_3 = T(3) * k4;
    const T k1_times_2 = T(2) * k1;
    const T k2_times_2 = T(2) * k2;
    const T k3_times_2 = T(2) * k3;

    const T &x0 = x_dist_norm, &y0 = y_dist_norm;
    T x = x0, y = y0;

    T d_x_dx = T(1);
    T d_x_dy = T(0);
    T d_y_dx = T(0);
    T d_y_dy = T(1);

    for (int i = 0; i < 5; ++i) {
        T r2          = x * x + y * y;
        T icdist      = T(1) / (T(1) + ((k4 * r2 + k1) * r2 + k0) * r2);
        T deltaX      = T(2) * k2 * x * y + k3 * (r2 + T(2) * x * x);
        T deltaY      = T(2) * k3 * x * y + k2 * (r2 + T(2) * y * y);
        T x0_minus_dx = x0 - deltaX;
        T y0_minus_dy = y0 - deltaY;

        // jacobian
        const T icdist2   = icdist * icdist;
        const T x_times_2 = T(2) * x;
        const T y_times_2 = T(2) * y;

        // d w.r.t x
        T d_r2           = x_times_2 * d_x_dx + y_times_2 * d_y_dx;
        T r2d_r2         = r2 * d_r2;
        T d_icdist       = -(k4_times_3 * r2 * r2d_r2 + k1_times_2 * r2d_r2 + k0 * d_r2) * icdist2;
        T d_xx_plus_d_yx = d_x_dx * y + d_y_dx * x;
        T d_dx           = k2_times_2 * d_xx_plus_d_yx + k3 * (d_r2 + T(2) * x_times_2 * d_x_dx);
        T d_dy           = k3_times_2 * d_xx_plus_d_yx + k2 * (d_r2 + T(2) * y_times_2 * d_y_dx);
        d_x_dx           = ((1 - d_dx) * icdist + x0_minus_dx * d_icdist);
        d_y_dx           = (-d_dy * icdist + y0_minus_dy * d_icdist);

        // d w.r.t y
        d_r2             = x_times_2 * d_x_dy + y_times_2 * d_y_dy;
        r2d_r2           = r2 * d_r2;
        d_icdist         = -(k4_times_3 * r2 * r2d_r2 + k1_times_2 * r2d_r2 + k0 * d_r2) * icdist2;
        T d_xy_plus_d_yy = d_x_dy * y + d_y_dy * x;
        d_dx             = k2_times_2 * d_xy_plus_d_yy + k3 * (d_r2 + T(2) * x_times_2 * d_x_dy);
        d_dy             = k3_times_2 * d_xy_plus_d_yy + k2 * (d_r2 + T(2) * y_times_2 * d_y_dy);
        d_x_dy           = (-d_dx * icdist + x0_minus_dx * d_icdist);
        d_y_dy           = ((1 - d_dy) * icdist + y0_minus_dy * d_icdist);

        x = x0_minus_dx * icdist;
        y = y0_minus_dy * icdist;
    }

    pt_undist_norm(0) = x;
    pt_undist_norm(1) = y;

    Eigen::Matrix<FloatType, 2, 2> Jdist;
    Jdist(0, 0) = d_x_dx;
    Jdist(0, 1) = d_y_dx;
    Jdist(1, 0) = d_x_dy;
    Jdist(1, 1) = d_y_dy;

    J = Jdist * Kinv_.template block<2, 2>(0, 0);
}

template<typename FloatType>
template<typename V1, typename V2>
inline void PinholeCameraModel<FloatType>::camera_to_undist_norm(const V1 &pt_c, V2 &pt_undist_norm) const {
    const FloatType invz = 1 / FloatType(pt_c(2));
    pt_undist_norm       = pt_c.template head<2>() * invz;
}

template<typename FloatType>
template<typename V1, typename V2>
inline void PinholeCameraModel<FloatType>::undist_norm_to_dist_norm(const V1 &pt_undist_norm, V2 &pt_dist_norm) const {
    const FloatType x  = pt_undist_norm(0);
    const FloatType y  = pt_undist_norm(1);
    const FloatType x2 = x * x;
    const FloatType y2 = y * y;
    const FloatType r2 = x2 + y2, r4 = r2 * r2, r6 = r2 * r4;
    const FloatType c_raddist = 1 + D_(0) * r2 + D_(1) * r4 + D_(4) * r6;
    pt_dist_norm(0)           = x * c_raddist + 2 * D_(2) * x * y + D_(3) * (r2 + 2 * x2);
    pt_dist_norm(1)           = y * c_raddist + D_(2) * (r2 + 2 * y2) + 2 * D_(3) * x * y;
}

template<typename FloatType>
template<typename V1, typename V2>
inline void PinholeCameraModel<FloatType>::norm_to_img(const V1 &pt_norm, V2 &pt_img) const {
    pt_img(0) = K_(0, 0) * pt_norm(0) + K_(0, 1) * pt_norm(1) + K_(0, 2);
    pt_img(1) = K_(1, 1) * pt_norm(1) + K_(1, 2);
}

template<typename FloatType>
template<typename V1, typename V2>
inline void PinholeCameraModel<FloatType>::img_to_norm(const V1 &pt_img, V2 &pt_norm) const {
    pt_norm(0) = Kinv_(0, 0) * pt_img(0) + Kinv_(0, 1) * pt_img(1) + Kinv_(0, 2);
    pt_norm(1) = Kinv_(1, 1) * pt_img(1) + Kinv_(1, 2);
}

template<typename FloatType>
template<typename V1, typename V2>
inline void PinholeCameraModel<FloatType>::dist_norm_to_undist_norm(const V1 &pt_dist_norm, V2 &pt_undist_norm) const {
    // Fixed-point iteration algorithm for point undistortion (from "opencv/trunk/modules/imgproc/src/undistort.cpp")
    // The idea of this method is based on the following observation:
    // Given the distorted point (x0,y0), we want to find the undistorted point such that
    // x0 = xu * raddist(xu,yu) + tandist(xu,yu) and y0 = yu * raddist(xu,yu) + tandist(xu,yu)
    // Denoting g(x,y) = [gx(x,y); gy(x,y)] = [(x0-tandist(x,y))/raddist(x,y); (y0-tandist(x,y))/raddist(x,y)],
    // we notice that (xu, yu) is a fixed-point of function g. We can therefore find this fixed-point as the limit
    // of the following recurrence: p_{n+1} = g(p_{n}), starting with p_{0} = (x0, y0)
    const FloatType x0 = pt_dist_norm(0);
    const FloatType y0 = pt_dist_norm(1);
    FloatType x        = x0;
    FloatType y        = y0;
    for (int i = 0; i < 5; ++i) {
        const FloatType xx         = x * x;
        const FloatType yy         = y * y;
        const FloatType r2         = xx + yy;
        const FloatType xx_times_3 = 3 * xx;
        const FloatType yy_times_3 = 3 * yy;
        const FloatType xy_times_2 = 2 * x * y;
        const FloatType icdist     = 1 / (1 + ((D_[4] * r2 + D_[1]) * r2 + D_[0]) * r2);
        const FloatType deltaX     = D_[2] * xy_times_2 + D_[3] * (yy + xx_times_3);
        const FloatType deltaY     = D_[2] * (xx + yy_times_3) + D_[3] * xy_times_2;
        x                          = (x0 - deltaX) * icdist;
        y                          = (y0 - deltaY) * icdist;
    }
    pt_undist_norm(0) = x;
    pt_undist_norm(1) = y;
}

} // namespace Metavision

#endif // METAVISION_SDK_CV_DETAIL_PINHOLE_CAMERA_MODEL_IMPL_H
