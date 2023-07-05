/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CALIBRATION_BLINKING_FRAME_GENERATOR_ALGORITHM_CONFIG_H
#define METAVISION_SDK_CALIBRATION_BLINKING_FRAME_GENERATOR_ALGORITHM_CONFIG_H

namespace Metavision {

/// @brief Structure to instantiate a Blinking Detector configuration.
struct BlinkingFrameGeneratorAlgorithmConfig {
    /// @brief Constructor
    /// @param accumulation_time Window of time during which events are considered to detect if a pixel is "blinking"
    /// @param min_num_blinking_pixels Minimum number of pixels needed to be detected before outputting a frame
    /// @param blinking_pixels_ratios_on The acceptable ratio of pixels that received only positive events
    /// over the number of pixels that received both during the accumulation window
    /// @param blinking_pixels_ratios_off The acceptable ratio of pixels that received only negative events
    /// over the number of pixels that received both during the accumulation window
    /// @param median_blur_radius Radius of the median blur applied on the mask of positive and negative pixels.
    /// a negative value disables the median blur. (diameter = 1 + 2*radius)
    /// @param enable_event_count Enable counting events instead of generating a binary mask.
    BlinkingFrameGeneratorAlgorithmConfig(double accumulation_time, int min_num_blinking_pixels = 100,
                                          double blinking_pixels_ratios_on  = 0.15,
                                          double blinking_pixels_ratios_off = 0.15, int median_blur_radius = 1,
                                          bool enable_event_count = false) :
        accumulation_time_(accumulation_time),
        min_num_blinking_pixels_(min_num_blinking_pixels),
        blinking_pixels_ratios_on_(blinking_pixels_ratios_on),
        blinking_pixels_ratios_off_(blinking_pixels_ratios_off),
        enable_event_count_(enable_event_count) {
        median_blur_diameter_ = (median_blur_radius < 0 ? 0 : 1 + 2 * median_blur_radius);
    }

    double accumulation_time_;
    int min_num_blinking_pixels_;
    double blinking_pixels_ratios_on_;
    double blinking_pixels_ratios_off_;

    unsigned int median_blur_diameter_; ///< Diameter of the median blur applied on the mask of positive and
                                        ///< negative pixels (diameter = 1 + 2*radius)
                                        ///< a zero value disables the median blur.

    bool enable_event_count_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CALIBRATION_BLINKING_FRAME_GENERATOR_ALGORITHM_CONFIG_H
