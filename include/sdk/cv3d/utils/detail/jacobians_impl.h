/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/
#ifndef METAVISION_SDK_CV3D_JACOBIANS_IMPL_H
#define METAVISION_SDK_CV3D_JACOBIANS_IMPL_H

namespace Metavision {

template<typename Scalar, int DIM>
void d_proj_point_d_xi(const Eigen::Matrix<Scalar, DIM, 1> &p, Eigen::Matrix<Scalar, 2, 6> &jac) {
    static_assert(std::is_floating_point<Scalar>::value, "Float type required");
    static_assert((DIM == 3 || DIM == 4), "p must either be a Vector3 or Vector4");

    // Precomputed multiplication of the jacobians for perspective projection of mapped point.
    // J = d{ persp( exp(xi)*pt3 ) } / d{ xi } = d{ persp(pt3) } / d{pt3} . d{ exp(xi)*pt3 } / d{ xi }
    const auto invz  = Scalar(1) / p.z();
    const auto xinvz = p.x() * invz;
    const auto yinvz = p.y() * invz;

    jac(0, 0) = invz;
    jac(0, 1) = 0;
    jac(0, 2) = -xinvz * invz;
    jac(0, 3) = -xinvz * yinvz;
    jac(0, 4) = 1 + xinvz * xinvz;
    jac(0, 5) = -yinvz;
    jac(1, 0) = 0;
    jac(1, 1) = invz;
    jac(1, 2) = -yinvz * invz;
    jac(1, 3) = -1 - yinvz * yinvz;
    jac(1, 4) = xinvz * yinvz;
    jac(1, 5) = xinvz;
}

template<typename Scalar, int DIMP, int DIMV>
void d_proj_vector_d_xi(const Eigen::Matrix<Scalar, DIMP, 1> &p, const Eigen::Matrix<Scalar, DIMV, 1> &v,
                        Eigen::Matrix<Scalar, 2, 6> &jac) {
    static_assert(std::is_floating_point<Scalar>::value, "Float type required");
    static_assert((DIMP == 3 || DIMP == 4), "p must either be a Vector3 or Vector4");
    static_assert((DIMV == 3 || DIMV == 4), "v must either be a Vector3 or Vector4");

    // Precomputed multiplication of the jacobians for perspective projection of mapped point.
    // J = lim {h -> 0} [ 1/h * (d{ persp( exp(xi)*(pt3+h*v3) ) } / d{ xi } - d{ persp( exp(xi)*pt3 ) } / d{ xi }) ]
    const auto invz   = Scalar(1) / p.z();
    const auto xinvz  = p.x() * invz;
    const auto yinvz  = p.y() * invz;
    const auto vxinvz = v.x() * invz;
    const auto vyinvz = v.y() * invz;
    const auto vzinvz = v.z() * invz;

    jac(0, 0) = -vzinvz * invz;
    jac(0, 1) = 0;
    jac(0, 2) = (2 * vzinvz * xinvz - vxinvz) * invz;
    jac(0, 3) = 2 * xinvz * yinvz * vzinvz - xinvz * vyinvz - yinvz * vxinvz;
    jac(0, 4) = 2 * xinvz * (xinvz * vzinvz - vxinvz);
    jac(0, 5) = vzinvz * yinvz - vyinvz;
    jac(1, 0) = 0;
    jac(1, 1) = -vzinvz * invz;
    jac(1, 2) = (2 * vzinvz * yinvz - vyinvz) * invz;
    jac(1, 3) = 2 * yinvz * (vyinvz - yinvz * vzinvz);
    jac(1, 4) = xinvz * vyinvz + yinvz * vxinvz - 2 * xinvz * yinvz * vzinvz;
    jac(1, 5) = vxinvz - vzinvz * xinvz;
}
} // namespace Metavision

#endif // METAVISION_SDK_CV3D_JACOBIANS_IMPL_H
