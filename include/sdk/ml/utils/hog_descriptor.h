/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_UTILS_HOG_DESCRIPTOR_H
#define METAVISION_SDK_ML_UTILS_HOG_DESCRIPTOR_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/ml/utils/rectangle_tools.h"

namespace Metavision {

/// @brief Class wrapping the OpenCV HogDescriptor module. It implements a border coping mechanism as well
class HOGDescriptor {
public:
    /// @brief Construct an HOGDescriptor object
    /// @param windowSize Size of the area to compute the HOG on.
    /// @param blocksize Size of the individual block to compute the HOG (corresponds to squares, e.g. a value of 16
    /// uses 16x16 squares).
    /// @param blockstride Space between two consecutive blocks. Using this parameter as less as the blocksize one could
    /// compute on overlapping blocks for instance.
    /// @param cellsize Size of the cells to compute gradient on. Must be less than blocksize.
    /// @param nbins Number of orientations chosen to discretize the HOG.
    HOGDescriptor(const cv::Size &windowSize, const cv::Size &blocksize, const cv::Size &blockstride,
                  const cv::Size &cellsize, int nbins) :
        hog_descriptor_(windowSize, blocksize, blockstride, cellsize, nbins), window_size_(windowSize){};
    ~HOGDescriptor(){};

    /// @brief Computes a Histogram Of Gradient (HOG)
    /// @param[in] img_to_describe Image representing the scene
    /// @param[in] roi Region Of Interest where the object is
    /// @param[out] descriptor Description of the object
    void compute_description(const cv::Mat &img_to_describe, const cv::Rect &roi, std::vector<float> &descriptor) {
        cv::Mat sub_img = RectTools::subwindow(img_to_describe, roi, cv::BORDER_CONSTANT);
        if (sub_img.size() != window_size_) {
            cv::resize(sub_img, sub_img, window_size_);
        }
        hog_descriptor_.compute(sub_img, descriptor);
    }

    float compute_similarity(const std::vector<float> &descriptor_1, const std::vector<float> &descriptor_2) {
        if (descriptor_1.size() != descriptor_2.size()) {
            MV_SDK_LOG_WARNING() << "Trying to evaluate difference between two descriptors with different sizes";
            return 0.f;
        }
        float denom_1 = 0;
        float denom_2 = 0;
        float dot     = 0;
        for (auto it1 = descriptor_1.cbegin(), it2 = descriptor_2.cbegin(); it1 != descriptor_1.cend(); it1++, it2++) {
            denom_1 += *it1 * (*it1);
            denom_2 += *it2 * (*it2);
            dot += *it1 * (*it2);
        }
        if ((denom_1 * denom_2) < 0.001f) {
            return 1.f;
        }
        return std::max(dot / (std::sqrt(denom_1) * std::sqrt(denom_2)), static_cast<float>(0.1));
    }

private:
    cv::HOGDescriptor hog_descriptor_;
    cv::Size window_size_;
};

} // namespace Metavision

#endif // METAVISION_SDK_ML_UTILS_HOG_DESCRIPTOR_H
