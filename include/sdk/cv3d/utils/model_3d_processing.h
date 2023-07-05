/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV3D_MODEL_3D_PROCESSING_H
#define METAVISION_SDK_CV3D_MODEL_3D_PROCESSING_H

#include <string>
#include <set>
#include <Eigen/Core>
#include <Eigen/LU>
#include <opencv2/opencv.hpp>

#include "metavision/sdk/cv3d/utils/edge_data_association.h"

namespace Metavision {

struct Model3d;

template<typename T>
class CameraGeometryBase;
using CameraGeometry32f = CameraGeometryBase<float>;

/// @brief Loads a 3D model from a JSON file
/// @param[in] path Path to the JSON file containing the 3D model
/// @param[out] model Structure containing the loaded 3D model if the function succeeds
/// @return True if the function succeeds, false otherwise
bool load_model_3d_from_json(const std::string &path, Model3d &model);

/// @brief Selects the visible edges of a 3D model given a camera's pose
///
/// This function assumes that the 3D model is a convex polyhedron whose faces are either completely hidden or visible
/// like explained in "Fast algorithm for 3D-graphics, Georg Glaeser, section 5.2".
///
/// @param[in] T_c_w Camera's pose from which visible edges need to be determined
/// @param[in] model The 3D model whose visible edges need to be determined
/// @param[out] visible_edges 3D model's edges visible from the given camera's pose
void select_visible_edges(const Eigen::Matrix4f &T_c_w, const Model3d &model, std::set<size_t> &visible_edges);

/// @brief Samples 3D support points from the visible edges of a 3D model
/// @param[in] cam_geometry Camera geometry instance allowing mapping points from world (i.e. the 3D model's reference
/// frame) to image coordinates
/// @param[in] T_c_w Camera's pose with respect to the 3D model
/// @param[in] model 3D model from which support points are to be sampled
/// @param[in] visible_edges 3D model's visible edges
/// @param[in] step_px Step in pixels between two support points in the distorted image
/// @param[out] edge_data_associations Edge data associations whose @p pt_w attribute will be filled with the sampled
/// support points
void sample_support_points(const CameraGeometry32f &cam_geometry, const Eigen::Matrix4f &T_c_w, const Model3d &model,
                           const std::set<size_t> &visible_edges, std::uint32_t step_px,
                           EdgeDataAssociationVector &edge_data_associations);

/// @brief Draws the selected edges of a 3D model into the output frame
/// @param cam_geometry Camera geometry instance allowing mapping points from world (i.e. the 3D model's reference
/// frame) to image coordinates
/// @param T_c_w Camera's pose from which edges need to be drawn
/// @param model 3D model whose edges need to be drawn
/// @param edges Indexes to the 3D model's edges that need to be drawn
/// @param output Output image
/// @param color Color used to render the 3D model's edges
void draw_edges(const CameraGeometry32f &cam_geometry, const Eigen::Matrix4f &T_c_w, const Model3d &model,
                const std::set<size_t> &edges, cv::Mat &output, const cv::Scalar &color = cv::Scalar(255, 0, 0));

} // namespace Metavision

#endif // METAVISION_SDK_CV3D_MODEL_3D_PROCESSING_H
