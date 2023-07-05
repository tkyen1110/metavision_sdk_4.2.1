/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV3D_EDGE_DATA_ASSOCIATION_H
#define METAVISION_SDK_CV3D_EDGE_DATA_ASSOCIATION_H

#include <Eigen/Core>
#include <Eigen/StdVector>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief 2D/3D association allowing tracking a 3D model from its edges
///
/// The association consists in a 3D point along one of the 3D model's edges and a 2D point in the undistorted
/// normalized image plane. The 3D model acts as a global coordinates system (i.e. the world).
struct EdgeDataAssociation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    size_t edge_idx;                   ///< Edge this data association is related to
    Eigen::Vector4f pt_w;              ///< The 3D point lying on one of the 3D model's edges
    Eigen::Vector4f dir_w;             ///< Direction of the 3D model's edge on which lies the 3D point
    Eigen::Vector2f match_undist_norm; ///< Match of the 3D point in undistorted normalized coordinates
    timestamp ts;                      ///< Timestamp of the event corresponding to the match
    float w = 1.f;                     ///< Weight associated with this association
};

/// @brief Buffer of @ref EdgeDataAssociation that uses Eigen's memory aligned allocator
using EdgeDataAssociationVector = std::vector<EdgeDataAssociation, Eigen::aligned_allocator<EdgeDataAssociation>>;
} // namespace Metavision

#endif // METAVISION_SDK_CV3D_EDGE_DATA_ASSOCIATION_H
