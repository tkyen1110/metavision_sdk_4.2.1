/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_DETAIL_OCAM_MODEL_IMPL_H
#define METAVISION_SDK_CV_DETAIL_OCAM_MODEL_IMPL_H

#include <Eigen/Dense>
#include <Eigen/LU>

#include "metavision/sdk/cv/utils/detail/vec_traits.h"
#include "metavision/sdk/cv/utils/detail/mat_traits.h"

namespace Metavision {

template<typename FloatType>
OcamModel<FloatType>::OcamModel(const Vec2i &img_size, const VecX &poly, const VecX &inv_poly, const Vec2 &center,
                                const Mat2CM &affine_transform, FloatType zoom_factor) {
    p_            = poly;
    i_p_          = inv_poly;
    center_       = center;
    A_            = affine_transform;
    i_A_          = A_.inverse();
    img_size_     = img_size;
    scale_        = std::abs(p_(0)) * zoom_factor;
    ideal_center_ = img_size_.template cast<FloatType>() * (FloatType(1) / 2);
}

template<typename FloatType>
const Eigen::Vector2i &OcamModel<FloatType>::get_image_size() const {
    return img_size_;
}

template<typename FloatType>
FloatType OcamModel<FloatType>::get_distance_to_image_plane() const {
    return scale_;
}

template<typename FloatType>
template<typename M>
void OcamModel<FloatType>::get_undist_norm_to_undist_img_transform(M &m) const {
    static_assert(mat_traits<M>::dimX == 3 && mat_traits<M>::dimY == 3, "Output matrix must be of dimensions 3x3");

    // clang-format off
    m(0, 0) = scale_      ; m(0, 1) = FloatType(0); m(0, 2) = ideal_center_(0);
    m(1, 0) = FloatType(0); m(1, 1) = scale_      ; m(1, 2) = ideal_center_(1);
    m(2, 0) = FloatType(0); m(2, 1) = FloatType(0); m(2, 2) = FloatType(1);
    // clang-format on
}

template<typename FloatType>
template<typename V1, typename V2>
inline void OcamModel<FloatType>::img_to_undist_norm(const V1 &pt_dist_img, V2 &pt_undist_norm) const {
    static_assert(vec_traits<V1>::dim == 2, "Input Vector must have 2 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");

    pt_undist_norm = i_A_ * pt_dist_img - center_;

    const FloatType r = pt_undist_norm.norm();

    FloatType z_p = p_(0);
    FloatType r_i = FloatType(1);

    for (typename VecX::Index i = 1; i < p_.size(); ++i) {
        r_i *= r;
        z_p += r_i * p_(i);
    }

    pt_undist_norm = pt_undist_norm / z_p;
}

template<typename FloatType>
template<typename V1, typename V2>
inline void OcamModel<FloatType>::camera_to_undist_img(const V1 &pt_c, V2 &pt_undist_img) const {
    static_assert(vec_traits<V1>::dim == 3, "Input Vector must have 3 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");

    pt_undist_img = pt_c.template head<2>();

    pt_undist_img *= scale_ / pt_c(2);
    pt_undist_img = pt_undist_img + ideal_center_;
}

template<typename FloatType>
template<typename V1, typename V2>
inline void OcamModel<FloatType>::undist_norm_to_undist_img(const V1 &pt_undist_norm, V2 &pt_undist_img) const {
    static_assert(vec_traits<V1>::dim == 2, "Input Vector must have 2 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");

    pt_undist_img = pt_undist_norm * scale_;
    pt_undist_img = pt_undist_img + ideal_center_;
}
template<typename FloatType>
template<typename V1, typename V2>
inline void OcamModel<FloatType>::undist_norm_to_dist_norm(const V1 &pt_undist_norm, V2 &pt_dist_norm) const {
    static_assert(vec_traits<V1>::dim == 2, "Input Vector must have 2 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");
    std::runtime_error("Not Implemented yet");
}

template<typename FloatType>
template<typename V1, typename V2>
inline void OcamModel<FloatType>::undist_img_to_undist_norm(const V1 &pt_undist_img, V2 &pt_undist_norm) const {
    static_assert(vec_traits<V1>::dim == 2, "Input Vector must have 2 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");

    pt_undist_norm = pt_undist_img - ideal_center_;
    pt_undist_norm *= (FloatType(1) / scale_);
}

template<typename FloatType>
template<typename V1, typename V2>
inline void OcamModel<FloatType>::undist_norm_to_img(const V1 &pt_undist_norm, V2 &pt_dist_img) const {
    static_assert(vec_traits<V1>::dim == 2, "Input Vector must have 2 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");

    camera_to_img_internal(pt_undist_norm, FloatType(1), pt_dist_img);
}

template<typename FloatType>
template<typename V1, typename V2>
inline void OcamModel<FloatType>::camera_to_img(const V1 &pt_c, V2 &pt_dist_img) const {
    static_assert(vec_traits<V1>::dim == 3, "Input Vector must have 3 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");

    const Vec2 xy_c = pt_c.template head<2>();

    camera_to_img_internal(xy_c, pt_c(2), pt_dist_img);
}

template<typename FloatType>
template<typename V1, typename V2, typename M>
inline void OcamModel<FloatType>::get_undist_norm_to_img_jacobian(const V1 &pt_undist_norm, V2 &pt_dist_img,
                                                                  M &J) const {
    static_assert(vec_traits<V1>::dim == 2, "Input Vector must have 2 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");
    static_assert(mat_traits<M>::dimX == 2 && mat_traits<M>::dimY == 2, "Output matrix must be of dimensions 2x2");

    const FloatType norm = pt_undist_norm.norm();

    if (norm > std::numeric_limits<FloatType>::epsilon()) {
        const FloatType norm2               = norm * norm;
        const FloatType inv_norm            = FloatType(1) / norm;
        const FloatType inv_norm3           = FloatType(1) / (norm2 * norm);
        const FloatType inv_norm3_plus_norm = FloatType(1) / (norm2 * norm + norm);
        const FloatType theta               = std::atan2(FloatType(1), norm);

        FloatType rho = i_p_(0);
        FloatType t_i = FloatType(1);

        for (typename VecX::Index i = 1; i < i_p_.size(); ++i) {
            t_i *= theta;
            rho += t_i * i_p_(i);
        }

        pt_dist_img = A_ * (pt_undist_norm * inv_norm * rho) + center_;

        const FloatType &xn  = pt_undist_norm(0);
        const FloatType &yn  = pt_undist_norm(1);
        const FloatType &a00 = A_(0, 0);
        const FloatType &a01 = A_(0, 1);
        const FloatType &a10 = A_(1, 0);
        const FloatType &a11 = A_(1, 1);

        // d uv w.r.t xn
        FloatType d_theta = -xn * inv_norm3_plus_norm;
        FloatType d_rho   = FloatType(0);

        FloatType t_i_minus_1 = FloatType(1); // theta^(i-1)
        for (typename VecX::Index i = 1; i < i_p_.size(); ++i) {
            d_rho += i_p_(i) * i * t_i_minus_1 * d_theta; // Warning: Factor by d_theta outside the loop
                                                          // leads to numerical issues in the gtest
            t_i_minus_1 *= theta;
        }

        // un = xn * rho / norm
        // vn = yn * rho / norm

        FloatType d_inv_norm = -xn * inv_norm3;
        FloatType d_un       = (inv_norm + d_inv_norm * xn) * rho + d_rho * xn * inv_norm;
        FloatType d_vn       = (d_inv_norm * yn) * rho + d_rho * yn * inv_norm;

        J(0, 0) = a00 * d_un + a01 * d_vn;
        J(1, 0) = a10 * d_un + a11 * d_vn;

        // d uv w.r.t yn
        d_theta = -yn * inv_norm3_plus_norm;
        d_rho   = FloatType(0);

        t_i_minus_1 = FloatType(1); // theta^(i-1)
        for (typename VecX::Index i = 1; i < i_p_.size(); ++i) {
            d_rho += i_p_(i) * i * t_i_minus_1 * d_theta; // Warning: Factor by d_theta outside the loop
                                                          // leads to numerical issues in the gtest
            t_i_minus_1 *= theta;
        }

        d_inv_norm = -yn * inv_norm3;
        d_un       = (d_inv_norm * xn) * rho + d_rho * xn * inv_norm;
        d_vn       = (inv_norm + d_inv_norm * yn) * rho + d_rho * yn * inv_norm;

        J(0, 1) = a00 * d_un + a01 * d_vn;
        J(1, 1) = a10 * d_un + a11 * d_vn;
    } else {
        J(0, 0) = FloatType(0);
        J(0, 1) = FloatType(0);
        J(1, 0) = FloatType(0);
        J(1, 1) = FloatType(0);

        pt_dist_img = center_;
    }
}

template<typename FloatType>
template<typename V1, typename V2, typename M>
inline void OcamModel<FloatType>::get_img_to_undist_norm_jacobian(const V1 &pt_dist_img, V2 &pt_undist_norm,
                                                                  M &J) const {
    static_assert(vec_traits<V1>::dim == 2, "Input Vector must have 2 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");
    static_assert(mat_traits<M>::dimX == 2 && mat_traits<M>::dimY == 2, "Output matrix must be of dimensions 2x2");

    pt_undist_norm = i_A_ * pt_dist_img - center_;

    const FloatType r = pt_undist_norm.norm();

    FloatType z_p = p_(0);
    FloatType r_i = FloatType(1);

    for (typename VecX::Index i = 1; i < p_.size(); ++i) {
        r_i *= r;
        z_p += r_i * p_(i);
    }

    const FloatType &x1      = pt_undist_norm(0);
    const FloatType &y1      = pt_undist_norm(1);
    const FloatType &a0      = i_A_(0, 0);
    const FloatType &a1      = i_A_(0, 1);
    const FloatType &a2      = i_A_(1, 0);
    const FloatType &a3      = i_A_(1, 1);
    const FloatType inv_zp_2 = FloatType(1) / (z_p * z_p);

    FloatType d_r_dx      = (x1 * a0 + y1 * a2) / r;
    FloatType d_r_dy      = (x1 * a1 + y1 * a3) / r;
    FloatType d_zp_dx     = FloatType(0);
    FloatType d_zp_dy     = FloatType(0);
    FloatType r_i_minus_1 = FloatType(1);

    for (typename VecX::Index i = 1; i < p_.size(); ++i) {
        d_zp_dx += i * r_i_minus_1 * d_r_dx * p_[i];
        d_zp_dy += i * r_i_minus_1 * d_r_dy * p_[i];
        r_i_minus_1 *= r;
    }

    J(0, 0) = (a0 * z_p - d_zp_dx * x1) * inv_zp_2;
    J(1, 0) = (a2 * z_p - d_zp_dx * y1) * inv_zp_2;
    J(0, 1) = (a1 * z_p - d_zp_dy * x1) * inv_zp_2;
    J(1, 1) = (a3 * z_p - d_zp_dy * y1) * inv_zp_2;

    pt_undist_norm = pt_undist_norm / z_p;
}

template<typename FloatType>
template<typename V1, typename V2>
inline void OcamModel<FloatType>::camera_to_img_internal(const V1 &pt_c, const FloatType &z, V2 &pt_dist_img) const {
    static_assert(vec_traits<V1>::dim == 2, "Input Vector must have 2 dimensions");
    static_assert(vec_traits<V2>::dim == 2, "Output Vector must have 2 dimensions");

    const FloatType norm = pt_c.norm();

    if (norm > std::numeric_limits<FloatType>::epsilon()) {
        const FloatType theta = std::atan2(z, norm);
        FloatType rho         = i_p_(0);
        FloatType t_i         = FloatType(1);

        for (typename VecX::Index i = 1; i < i_p_.size(); ++i) {
            t_i *= theta;
            rho += t_i * i_p_(i);
        }

        const FloatType i_norm = FloatType(1) / norm;

        pt_dist_img = A_ * (pt_c * i_norm * rho) + center_;

    } else {
        pt_dist_img = center_;
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CV_DETAIL_OCAM_MODEL_IMPL_H
