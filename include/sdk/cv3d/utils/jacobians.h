/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/
#ifndef METAVISION_SDK_CV3D_JACOBIANS_H
#define METAVISION_SDK_CV3D_JACOBIANS_H

#include <Eigen/Core>

namespace Metavision {

/// @brief Computes the Jacobian matrix J with respect to the SE3 pose increment xi=[vx,vy,vz,wx,wy,wz] (where linear
/// velocity v=[vx,vy,vz] and angular velocity w=[wx,wy,wz]) for the mapped-3D-point-perspective-projection operation
/// 'persp( exp(xi)*pt3 )'.
/// @tparam Scalar Either float or double
/// @tparam DIM Dimensions of the input point, either 3 or 4
/// @param[in] p 3D point to project
/// @param[out] jac Operation's Jacobian
template<typename Scalar, int DIM>
void d_proj_point_d_xi(const Eigen::Matrix<Scalar, DIM, 1> &p, Eigen::Matrix<Scalar, 2, 6> &jac);

/// @brief Computes the Jacobian matrix J with respect to the SE3 pose increment xi=[vx,vy,vz,wx,wy,wz] (where linear
/// velocity v=[vx,vy,vz] and angular velocity w=[wx,wy,wz]) for the mapped-3D-local-vector-perspective-projection
/// operation 'persp( exp(xi)*v3 @ exp(xi)*pt3 )'
/// @tparam Scalar Either float or double
/// @tparam DIMP Dimension of the input point, either 3 or 4
/// @tparam DIMV Dimension of the input vector, either 3 or 4
/// @param[in] p 3D point from which the vector is projected
/// @param[in] v Vector to project
/// @param[out] jac Operation's Jacobian
/// @note This Jacobian does not include an eventual normalization of the vector after projection
template<typename Scalar, int DIMP, int DIMV>
void d_proj_vector_d_xi(const Eigen::Matrix<Scalar, DIMP, 1> &p, const Eigen::Matrix<Scalar, DIMV, 1> &v,
                        Eigen::Matrix<Scalar, 2, 6> &jac);
} // namespace Metavision

#include "metavision/sdk/cv3d/utils/detail/jacobians_impl.h"

#endif // METAVISION_SDK_CV3D_JACOBIANS_H