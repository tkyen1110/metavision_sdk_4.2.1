/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_HEAT_MAP_FRAME_GENERATOR_ALGORITHM_H
#define METAVISION_SDK_ANALYTICS_HEAT_MAP_FRAME_GENERATOR_ALGORITHM_H

#include <memory>
#include <opencv2/opencv.hpp>

namespace Metavision {

class CvColorMap;

/// @brief Class that produces a BGR image of a floating point value map
///
/// A colormap bar at the bottom of the image shows the color convention. Pixels for which a value was not computed
/// are shown in black. Also pixels outside the defined minimum/maximum values are shown in black too.
class HeatMapFrameGeneratorAlgorithm {
public:
    /// @brief Type of floating point values
    using float_type = float;

    /// @brief Constructor
    /// @param min_value Minimum value
    /// @param max_value Maximum value
    /// @param value_precision Precision used to determine the number of decimal digits to display (0.5, 0.01, ...)
    /// @param width Sensor's width (in pixels)
    /// @param height Sensor's height (in pixels)
    /// @param unit_str String representation of the unit of measurement, e.g. Hz, us, m/s ...
    /// @param cmap Colormap used to colorize the value map
    /// @param do_invert_cmap Flip the uchar values before applying the color map
    HeatMapFrameGeneratorAlgorithm(float_type min_value, float_type max_value, float_type value_precision, int width,
                                   int height, const std::string &unit_str = std::string(),
                                   unsigned int cmap = cv::COLORMAP_JET, bool do_invert_cmap = true);

    /// @brief Destructor
    ~HeatMapFrameGeneratorAlgorithm();

    /// @brief Draws the value map with a bar showing the colormap convention
    /// @param value_map Input value map, 1 floating point channel (CV_32FC1)
    /// @param out_image_bgr Output image
    void generate_bgr_heat_map(const cv::Mat_<float_type> &value_map, cv::Mat &out_image_bgr);

    /// @brief Returns the full generated image's width
    int get_full_width() const;

    /// @brief Returns the full generated image's height
    int get_full_height() const;

private:
    /// @brief Applies colormap on the grayscale value map and returns the composed output image with colormap bar
    /// @param gray_value_map Value map rescaled to 8 bits
    /// @param mask_valid_values Mask keeping valid values
    /// @param out_image_bgr Output image
    void apply_colormap_and_draw(const cv::Mat_<uchar> &gray_value_map, const cv::Mat_<uchar> &mask_valid_values,
                                 cv::Mat &out_image_bgr);

    // Minimum/maximum values
    float_type min_value_{0.f};
    float_type max_value_{0.f};

    unsigned int value_string_precision_{0}; ///< Number of decimal places needed for value_precision_

    // Size of the output image
    int width_{0};
    int height_{0};

    // Value map rescaling
    float_type rescaling_alpha_;
    float_type rescaling_beta_;

    // Color Map
    std::unique_ptr<CvColorMap> color_map_; ///< Colormap used to colorize the value map
    bool do_invert_cmap_;                   ///< Flip the uchar values before applying the color map

    const std::string unit_str_; ///< String representation of the unit of measurement, e.g. Hz, us, m/s ...
    cv::Vec3b text_color_ = cv::Vec3b::all(255);

    // Auxiliary images
    cv::Mat colormap_bar_;     ///< Bar to draw at the bottom of the image
    cv::Mat value_map_scaled_; ///< Input value map rescaled to 8 bits
    cv::Mat heat_map_colored_; ///< Colorized value map

    cv::Rect value_map_roi_; ///< ROI pointing to the value map part of output_img_
    cv::Rect colorbar_roi_;  ///< ROI pointing to the colormap part of output_img_

    // Height of the colormap bar.
    static constexpr int ColorMapHeight_ = 20;
    static constexpr int TextShift_      = 7; ///< Horizontal/Vertical gap in pixels between the image
                                              ///< edge and displayed text

    static constexpr unsigned char MinIntensity_ = 35;
    static constexpr unsigned char MaxIntensity_ = 215;
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_HEAT_MAP_FRAME_GENERATOR_ALGORITHM_H
