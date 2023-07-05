/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_UTILS_RECTANGLE_TOOLS_H
#define METAVISION_SDK_ML_UTILS_RECTANGLE_TOOLS_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

namespace Metavision {

namespace {

/// @brief Returns the first column index outside of the rectangle
/// @param rect Sub-rectangle of an OpenCV matrix
template<typename t>
inline t col_end(const cv::Rect_<t> &rect) {
    return rect.x + rect.width;
}

/// @brief Returns the first row index outside of the rectangle
/// @param rect Sub-rectangle of an OpenCV matrix
template<typename t>
inline t row_end(const cv::Rect_<t> &rect) {
    return rect.y + rect.height;
}

/// @brief Crops a rectangle to fit into the overall limits
/// @param[in,out] rect OpenCV rectangle to be cropped
/// @param[in] limit OpenCV rectangle in which the rectangle should fit
template<typename t>
inline void crop(cv::Rect_<t> &rect, const cv::Rect_<t> &limit) {
    if (rect.x + rect.width > limit.x + limit.width)
        rect.width = (limit.x + limit.width - rect.x);
    if (rect.y + rect.height > limit.y + limit.height)
        rect.height = (limit.y + limit.height - rect.y);
    if (rect.x < limit.x) {
        rect.width -= (limit.x - rect.x);
        rect.x = limit.x;
    }
    if (rect.y < limit.y) {
        rect.height -= (limit.y - rect.y);
        rect.y = limit.y;
    }
    if (rect.width < 0)
        rect.width = 0;
    if (rect.height < 0)
        rect.height = 0;
}

/// @brief Crops a rectangle to fit into the overall limits
/// @param[in,out] rect OpenCV rectangle to be cropped
/// @param[in] width Width of the boundary box
/// @param[in] height Height of the boundary box
/// @param[in] x X-base position of the boundary box
/// @param[in] y Y-base position of the boundary box
template<typename t>
inline void crop(cv::Rect_<t> &rect, t width, t height, t x = 0, t y = 0) {
    const cv::Rect_<t> bounds(x, y, width, height);
    crop(rect, bounds);
}

/// @brief Number of pixels that compound the image border
template<typename t>
struct Border {
    t top;    ///< Number of border pixels on the top of image
    t bottom; ///< Number of border pixels on the bottom of image
    t left;   ///< Number of border pixels on the left of image
    t right;  ///< Number of border pixels on the right of image
};

/// @brief Computes missing pixels in each direction in a sub-image
/// @param original Full original bounds
/// @param limited Cropped bounds
/// @return number of pixels removed in each directions
template<typename t>
inline Border<t> getBorder(const cv::Rect_<t> &original, const cv::Rect_<t> &limited) {
    Border<t> res;
    res.left   = limited.x - original.x;
    res.top    = limited.y - original.y;
    res.right  = col_end(original) - col_end(limited);
    res.bottom = row_end(original) - row_end(limited);
    assert(res.left >= 0 && res.top >= 0 && res.right >= 0 && res.bottom >= 0);
    return res;
}
} // namespace

/// @brief Helpers for computing sub-image of predefined fixed size from any position
namespace RectTools {

/// @brief Extracts a copy of the subimage and adds border if parts of the subimage are missing
/// @param in Input image from which the subimage is extracted
/// @param window Description of the subimage bounds
/// @param borderType Method to generate the borders
/// @return A copy of the subimage with its borders if required
inline cv::Mat subwindow(const cv::Mat &in, const cv::Rect &window, int borderType = cv::BORDER_CONSTANT) {
    cv::Rect cutWindow = window;
    crop(cutWindow, in.cols, in.rows);
    auto border = getBorder(window, cutWindow);
    // check if borders are missing
    if (border.left > 0 || border.top > 0 || border.right > 0 || border.bottom > 0) {
        cv::Mat subimage = in(cutWindow);
        cv::Mat output(subimage.rows + border.top + border.right, subimage.cols + border.left + border.right,
                       subimage.type());
        cv::copyMakeBorder(subimage, output, border.top, border.bottom, border.left, border.right,
                           borderType | cv::BORDER_ISOLATED);
        return output;
    } else {
        cv::Mat res = in(cutWindow).clone();
        return res;
    }
}

} // namespace RectTools
} // namespace Metavision

#endif // METAVISION_SDK_ML_UTILS_RECTANGLE_TOOLS_H
