/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CALIBRATION_CALIBRATE_MONO_CAMERA_H
#define METAVISION_SDK_CALIBRATION_CALIBRATE_MONO_CAMERA_H

#include <vector>
#include <opencv2/core/types.hpp>

namespace Metavision {
namespace MonoCalibration {

enum Model : int {
    Pinhole,
    Fisheye,
};

/// @brief Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern (using
/// OpenCV)
/// @param pattern_3d [in] 3D coordinates of the calibration pattern
/// @param pts_2d [in] 2D detections of each calibration pattern view
/// @param image_size [in] Sensor's size (in pixels)
/// @param model [in] Camera model
/// @param flags [in] Calibration flags
/// @param K [out] Camera matrix
/// @param d [out] Distortion coefficients
/// @param selected_views [out] Optional output bitset indicating the views that have been selected for the calibration
/// @param rvecs [out] Optional output vector of rotation vectors (Rodrigues) estimated for each pattern view
/// @param tvecs [out] Optional output vector of translation vectors estimated for each pattern view
/// @param outlier_ths [in] Remove the views for which the reprojection error is more than a certain number of times the
/// standard deviation away from the mean. For instance, a value of 2 means that views with error above (mean+2*std) are
/// removed. A negative threshold can be used to specify that all views must be kept
template<typename T>
T calibrate_opencv(const std::vector<cv::Point3_<T>> &pattern_3d, const std::vector<std::vector<cv::Point_<T>>> &pts_2d,
                   const cv::Size &image_size, MonoCalibration::Model model, int flags, cv::Mat &K, cv::Mat &d,
                   std::vector<bool> *selected_views = nullptr, std::vector<cv::Vec3d> *rvecs = nullptr,
                   std::vector<cv::Vec3d> *tvecs = nullptr, float outlier_ths = 2);

/// @brief Projects 2D points using camera intrinsic and extrinsic parameters for a given view (using OpenCV)
/// @param pattern_3d [in] 3D coordinates of the calibration pattern
/// @param K [in] Camera matrix
/// @param d [in] Distortion coefficients
/// @param rvec [in] Rotation vector (Rodrigues) of the input view
/// @param tvec [in] Translation vector of the input view
/// @param model [in] Camera model corresponding to @p K and @p d
/// @param projected_pts_2d [out] 2D projections of the pattern from the input view
template<typename T>
void project_points_opencv(const std::vector<cv::Point3_<T>> &pattern_3d, const cv::Mat &K, const cv::Mat &d,
                           const cv::Vec3d &rvec, const cv::Vec3d &tvec, MonoCalibration::Model model,
                           std::vector<cv::Point_<T>> &projected_pts_2d);

/// @brief Computes the average reprojection error of each input view given camera intrinsic and extrinsic parameters
/// (using OpenCV)
/// @param pattern_3d [in] 3D coordinates of the calibration pattern
/// @param pts_2d [in] 2D detections of each calibration pattern view
/// @param K [in] Camera matrix
/// @param d [in] Distortion coefficients
/// @param selected_views [int] Bitset indicating the views that been selected for the calibration
/// @param rvecs [in] Vector of rotation vectors (Rodrigues) estimated for each pattern view
/// @param tvecs [in] Vector of translation vectors estimated for each pattern view
/// @param model [in] Camera model corresponding to @p K and @p d
/// @param rms_errors [out] RMS reprojection errors of each input view
template<typename T>
void compute_reprojection_errors_opencv(const std::vector<cv::Point3_<T>> &pattern_3d,
                                        const std::vector<std::vector<cv::Point_<T>>> &pts_2d, const cv::Mat &K,
                                        const cv::Mat &d, const std::vector<bool> &selected_views,
                                        const std::vector<cv::Vec3d> &rvecs, const std::vector<cv::Vec3d> &tvecs,
                                        MonoCalibration::Model model, std::vector<float> &rms_errors);

} // namespace MonoCalibration
} // namespace Metavision

#endif // METAVISION_SDK_CALIBRATION_CALIBRATE_MONO_CAMERA_H
