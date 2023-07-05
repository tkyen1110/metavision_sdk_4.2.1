/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// This code sample demonstrates how to use the Metavision SDK CV's @ref CameraGeometry class to undistort and distort
// points. More specifically, this sample shows how this class can directly be used when dealing with Eigen types or how
// it can be used in combination with helpers when dealing with different ones.

#include <opencv2/opencv.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/cv/utils/camera_geometry_helpers.h>
#include <metavision/sdk/cv/utils/camera_geometry.h>
#include <metavision/sdk/cv/utils/pinhole_camera_model.h>

#include "constants.h"

int main(int argc, char *argv[]) {
    using PinholeModel32f    = Metavision::PinholeCameraModel<float>;
    using PinholeGeometry32f = Metavision::CameraGeometry<PinholeModel32f>;

    // Initialize the camera geometry.
    Eigen::Vector2i img_size;
    img_size << 640, 400;

    const std::vector<float> K = {411.6194f, 0.f, 326.9067f, 0.f, 411.9001f, 191.5793f, 0.f, 0.f, 1.f};
    const std::vector<float> D = {-0.3968f, 0.1457f, -5.5551e-04f, -0.0010f, -0.0244f};
    PinholeModel32f pinhole_model(640, 400, K, D);
    PinholeGeometry32f cam_geometry(pinhole_model);

    // Initialize the distortion maps used to undistort an image.
    cv::Mat_<float> mapx, mapy;
    mapx.create(cv::Size(img_size(0), img_size(1)));
    mapy.create(cv::Size(img_size(0), img_size(1)));

    Metavision::get_distortion_maps(cam_geometry, mapx, mapy);

    // Undistort an image.
    auto img = cv::imread(file_path);

    cv::Mat img_undist;
    cv::remap(img, img_undist, mapx, mapy, cv::INTER_LINEAR);

    cv::Mat himg;
    cv::hconcat(img, img_undist, himg);
    cv::imshow("distorted - undistorted", himg);
    cv::waitKey(0);

    // Direct use of the CameraGeometryBase class to undistort points represented by Eigen types.
    // We can use vectors or matrices to store the points.
    Eigen::Vector2f eigen_dist_img_pt;
    Eigen::Matrix2f eigen_undist_pts;
    eigen_dist_img_pt << 320, 200;

    cam_geometry.img_to_undist_norm(eigen_dist_img_pt, eigen_undist_pts.block<2, 1>(0, 0));
    cam_geometry.undist_norm_to_undist_img(eigen_undist_pts.block<2, 1>(0, 0), eigen_undist_pts.block<2, 1>(0, 1));

    MV_LOG_INFO() << "Using Eigen types:";
    MV_LOG_INFO() << "Undistorted normalized point:" << eigen_undist_pts.block<2, 1>(0, 0).transpose();
    MV_LOG_INFO() << "Undistorted image point:" << eigen_undist_pts.block<2, 1>(0, 1).transpose();

    // Use the helpers to undistort points represented by types different from the Eigen ones
    cv::Point2f cv_dist_img_pt = {320, 200};
    cv::Matx21f cv_undist_norm_pt;
    std::vector<float> std_undist_img_pt(2);

    Metavision::img_to_undist_norm(cam_geometry, cv_dist_img_pt, cv_undist_norm_pt);
    Metavision::undist_norm_to_undist_img(cam_geometry, cv_undist_norm_pt, std_undist_img_pt);

    MV_LOG_INFO() << "\nUsing different types:";
    MV_LOG_INFO() << "Undistorted normalized point:" << cv_undist_norm_pt;
    MV_LOG_INFO() << "Undistorted image point: [" << std_undist_img_pt[0] << "," << std_undist_img_pt[1] << "]";

    return 0;
}