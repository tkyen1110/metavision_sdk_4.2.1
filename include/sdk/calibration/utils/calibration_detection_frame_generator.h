/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CALIBRATION_CALIBRATION_DETECTION_FRAME_GENERATOR_H
#define METAVISION_SDK_CALIBRATION_CALIBRATION_DETECTION_FRAME_GENERATOR_H

#include <vector>
#include <opencv2/core/mat.hpp>

namespace Metavision {

/// @brief Structure to store the pattern's keypoints (static chessboard, blinking chessboard,
/// blinking points grid...) and the frame from which they were extracted.
struct CalibrationDetectionResult {
    /// @brief Default constructor
    CalibrationDetectionResult() = default;

    /// @brief Move operator
    CalibrationDetectionResult(CalibrationDetectionResult &&other);

    /// @brief Move assignment operator
    CalibrationDetectionResult &operator=(CalibrationDetectionResult &&other);

    /// @brief Resets the frame and the keypoints
    /// @param width Image's width
    /// @param height Image's height
    /// @param channels Number of color channels
    void reset(int width, int height, int channels = CV_8UC3);

    std::vector<cv::Point2f> keypoints_; ///< Calibration pattern's keypoints
    cv::Mat frame_;                      ///< Frame on which the keypoints have been extracted
};

/// @brief Class that generates an image used for calibration
///
///  Generates an image by combining:
///  - the image of a calibration pattern (either frame-based image of a static pattern, or event-based accumulation
///    image of a blinking pattern).
///  - the vector containing the pattern's keypoints.
///  - a colored overlay that shows which regions have been well covered during the calibration.
class CalibrationDetectionFrameGenerator {
public:
    /// @brief Type of pattern used during the calibration. It's required to be able to draw the pattern correctly.
    enum PatternMode {
        Chessboard ///< The pattern is a cartesian grid
    };

    /// @brief Constructor. It assumes a calibration pattern aligned according to a grid, distorted or not, so that
    /// the concept of rows and columns make sense to describe the pattern.
    /// @param width Image's width
    /// @param height Image's height
    /// @param cols Number of cols in the pattern (its width)
    /// @param rows Number of rows of the pattern (its height)
    /// @param pattern_mode Type of pattern used during the calibration (@ref PatternMode)
    /// @param overlay_convex_hull If true, overlay a "shadow" over the convex hull covering all the dots in the
    /// pattern. If false, draw only a circle over each dot.
    /// @param four_corners_only Extract only the 4 extreme corners to compute the pattern's convex hull used for the
    /// colored overlay (assuming there's no distortion and the grid looks like a rectangle). This has no effect if
    /// @p overlay_convex_hull is false.
    CalibrationDetectionFrameGenerator(unsigned int width, unsigned int height, unsigned int cols, unsigned int rows,
                                       PatternMode pattern_mode = PatternMode::Chessboard,
                                       bool overlay_convex_hull = false, bool four_corners_only = false);

    /// @brief Generates an image representing a calibration pattern on top of the frame on which it has been extracted.
    /// There's also a colored overlay to show which regions have been well covered during the calibration.
    /// @param output_img Output image
    /// @param results Structure containing a vector of keypoints and the frame on which they were extracted
    void generate_bgr_img(cv::Mat &output_img, const CalibrationDetectionResult &results);

private:
    /// @brief Extracts the keypoints along the contour of the grid to get the convex hull.
    /// @param input_points Pattern's keypoints
    /// @param output_hull Peripheral keypoints that wrap around the pattern
    /// @return false if there is not the right number of points
    bool get_pattern_hull(const std::vector<cv::Point2f> &input_points, std::vector<cv::Point> &output_hull);

    PatternMode pattern_mode_;
    unsigned int width_, height_;
    unsigned int cols_, rows_;
    unsigned int n_pts_;
    cv::Size grid_size_;

    int pattern_counts_; ///< Number of patterns collected so far

    bool overlay_convex_hull_;    ///< Overlay the convex hull. If false, only draw circles around each pattern point.
    bool hull_four_corners_only_; ///< Use only the 4 extreme corners to estimate the pattern's convex hull
    int hull_size_;               ///< Either 4 or  2 * (cols_ - 1 + rows_ - 1)
    std::vector<cv::Point> hull_pts_;
    cv::Mat crt_mask_;
    cv::Mat overlay_mask_; ///< Buffer image overlaying previous pattern positions

    static const cv::Vec3b kHullOverlayColorIncrement;  ///< Color increment to add to a pixel at
                                                        ///< each presence of the pattern
                                                        ///< (when drawing the hull).
    static const cv::Vec3b kPointOverlayColorIncrement; ///< Color increment to add to a pixel at
                                                        ///< each presence of the pattern.
    static const cv::Vec3b kTextColor;
};

} // namespace Metavision

#endif // METAVISION_SDK_CALIBRATION_CALIBRATION_DETECTION_FRAME_GENERATOR_H
