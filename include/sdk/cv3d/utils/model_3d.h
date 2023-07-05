/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV3D_MODEL_3D_H
#define METAVISION_SDK_CV3D_MODEL_3D_H

#include <utility>
#include <vector>
#include <Eigen/Core>

namespace Metavision {

/// @brief Structure defining a 3D model
struct Model3d {
    /// @brief Structure defining an edge, which consists in two indexes to model's vertices
    struct Edge {
        size_t tail;
        size_t head;
    };

    /// @brief Structure defining 3D model's face
    struct Face {
        std::vector<size_t> edges_indexes_; ///< Indexes to the model's edges that form this face
        Eigen::Vector4f normal_;            ///< Face's normal
    };

    std::vector<Eigen::Vector3f> vertices_; ///< All the vertices forming the 3D model
    std::vector<Edge> edges_; ///< All the edges forming the 3D model's faces. An edge consists in two indexes to
                              ///< model's vertices
    std::vector<Face> faces_; ///< All the faces forming the 3D model
};
} // namespace Metavision

#endif // METAVISION_SDK_CV3D_MODEL_3D_H
