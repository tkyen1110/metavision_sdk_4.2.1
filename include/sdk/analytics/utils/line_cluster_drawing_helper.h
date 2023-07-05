/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_LINE_CLUSTER_DRAWING_HELPER_H
#define METAVISION_SDK_ANALYTICS_LINE_CLUSTER_DRAWING_HELPER_H

#include <opencv2/core/mat.hpp>

namespace Metavision {

/// @brief Class that superimposes event-clusters on the horizontal lines drawn on an image filled with events
class LineClusterDrawingHelper {
public:
    LineClusterDrawingHelper()  = default;
    ~LineClusterDrawingHelper() = default;

    /// @brief Draws colored segments along an horizontal line
    /// @tparam InputIt Iterator to a cluster such as @ref LineClusterWithId. Required class members are x_begin,
    /// x_end and id. The ID refers to the ordinate of the line cluster
    /// @param output_img Output image
    /// @param first First line cluster to display
    /// @param last Last line cluster to display
    template<typename InputIt>
    inline void draw(cv::Mat &output_img, InputIt first, InputIt last) {
        for (auto it = first; it != last; ++it) {
            cv::Vec3b *row_ptr = output_img.ptr<cv::Vec3b>(static_cast<int>(it->id));
            // Draw a colored segment along the line for each cluster
            for (int x = it->x_begin; x <= it->x_end; x++) {
                if (x < 0 || x >= output_img.cols)
                    continue;
                row_ptr[x] = color_line_cluster_;
            }
        }
    }

private:
    const cv::Vec3b color_line_cluster_ = cv::Vec3b(0, 255, 255);
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_LINE_CLUSTER_DRAWING_HELPER_H
