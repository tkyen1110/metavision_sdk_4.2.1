/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_SPARSE_FLOW_FRAME_GENERATOR_ALGORITHM_H
#define METAVISION_SDK_CV_SPARSE_FLOW_FRAME_GENERATOR_ALGORITHM_H

#include <opencv2/opencv.hpp>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "metavision/sdk/core/utils/colors.h"
#include "metavision/sdk/cv/events/event_optical_flow.h"

namespace Metavision {

class SparseFlowFrameGeneratorAlgorithm {
public:
    /// @brief Stores one motion arrow per centroid (several optical flow events may have the same centroid) in the
    /// motion arrow map to be displayed later using the @ref update_frame_with_flow method.
    template<typename InputIt>
    void add_flow_for_frame_update(InputIt first, InputIt last) {
        using input_type = typename std::iterator_traits<InputIt>::value_type;

        static_assert(std::is_same<Metavision::EventOpticalFlow, input_type>::value, "Expected an EventOpticalFlow");

        if (std::distance(first, last) <= 0) {
            return;
        }

        // Goes from end to begin to get the latest flow event for a given centroid
        last = std::prev(last);
        for (; first != last; --last) {
            if (ids_.count(last->id) == 0) {
                CentroidFlowDisplay &motion_arrow = centroids_flow_map_[last->id];
                motion_arrow.centroid_            = cv::Point(std::lround(last->center_x), std::lround(last->center_y));
                cv::Point velocity(std::lround(last->vx / 4), std::lround(last->vy / 4));
                motion_arrow.bot_ = motion_arrow.centroid_ + velocity;

                cv::Mat magnitude, angle;
                cv::cartToPolar(static_cast<double>(velocity.x), static_cast<double>(velocity.y), magnitude, angle,
                                true);

                motion_arrow.color_ =
                    Metavision::hsv2rgb({angle.at<double>(0), 1.0, std::min(1., magnitude.at<double>(0) / 30.)});
                ids_.insert(last->id);
            }
        }
    }

    void clear_ids() {
        ids_.clear();
    }

    /// @brief Updates the input frame with the centroids' motion stored in the history
    ///
    /// Clears the history afterwards
    void update_frame_with_flow(cv::Mat &display_mat) {
        for (const auto &motion_arrow : centroids_flow_map_) {
            cv::arrowedLine(display_mat, motion_arrow.second.centroid_, motion_arrow.second.bot_,
                            cv::Scalar(motion_arrow.second.color_.r * 255, motion_arrow.second.color_.g * 255,
                                       motion_arrow.second.color_.b * 255),
                            2);
        }
        centroids_flow_map_.clear();
    }

private:
    /// @brief Small struct that holds display information for centroids
    struct CentroidFlowDisplay {
        cv::Point centroid_, bot_;
        Metavision::RGBColor color_;
    };

    std::unordered_set<unsigned int> ids_;
    std::unordered_map<uint32_t, CentroidFlowDisplay> centroids_flow_map_;
};

using FlowFrameGeneratorAlgorithm
    [[deprecated("This alias is deprecated since version 4.1.0 and will be removed in future releases")]] =
        SparseFlowFrameGeneratorAlgorithm;

} // namespace Metavision

#endif // METAVISION_SDK_CV_SPARSE_FLOW_FRAME_GENERATOR_ALGORITHM_H
