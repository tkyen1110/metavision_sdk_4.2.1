/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_ALGORITHMS_NON_MAXIMUM_SUPPRESSION_H
#define METAVISION_SDK_ML_ALGORITHMS_NON_MAXIMUM_SUPPRESSION_H

#include <fstream>
#include <iterator>
#include <algorithm>
#include <list>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/exceptions.hpp>
#include <boost/property_tree/info_parser.hpp>

#include <opencv2/opencv.hpp>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/core/utils/detail/iterator_traits.h"
#include "metavision/sdk/core/events/event_bbox.h"

namespace Metavision {

/// @brief Rescales events from network input format to the sensor's size and suppresses Non-Maximum overlapping boxes
class NonMaximumSuppressionWithRescaling {
public:
    /// @brief Builds non configured NonMaximumSuppressionWithRescaling object
    NonMaximumSuppressionWithRescaling() {}

    /// @brief Constructs object that rescales detected boxes and suppresses Non-Maximum overlapping boxes
    /// @param num_classes Number of possible class returned by neural network
    /// @param events_input_width Sensor's width
    /// @param events_input_height Sensor's height
    /// @param network_input_width Neural network input frame's width
    /// @param network_input_height Neural network input frame's height
    /// @param iou_threshold Threshold on IOU metrics to consider that two boxes are matching
    NonMaximumSuppressionWithRescaling(std::size_t num_classes, int events_input_width, int events_input_height,
                                       int network_input_width, int network_input_height, float iou_threshold) :
        num_classes_(num_classes),
        events_input_width_(events_input_width),
        events_input_height_(events_input_height),
        network_input_width_(network_input_width),
        network_input_height_(network_input_height),
        output_width_scaling_(static_cast<float>(events_input_width) / network_input_width),
        output_height_scaling_(static_cast<float>(events_input_height) / network_input_height_),
        iou_threshold_(iou_threshold),
        valid_classes_(num_classes, true) {}

    /// @brief Rescales and filters boxes
    /// @tparam InputIt Read-Only input iterator type
    /// @tparam OutputIt Read-Write output iterator type
    /// @param it_begin Iterator to the first box
    /// @param it_end Iterator to the past-the-end box
    /// @param inserter Output iterator or back inserter
    template<typename InputIt, typename OutputIt>
    void process_events(const InputIt it_begin, const InputIt it_end, OutputIt inserter) {
        assert(num_classes_ > 0);
        std::vector<std::list<EventBbox>> bbox_per_class(num_classes_);
        for (auto it = it_begin; it != it_end; ++it) {
            assert(it->class_id < num_classes_);
            bbox_per_class[it->class_id].push_back(*it);
        }

        for (std::size_t i = 0; i < num_classes_; ++i) {
            if (valid_classes_[i] == false) {
                continue;
            }
            NonMaximumSuppressionWithRescaling::compute_nms_per_class(bbox_per_class[i], iou_threshold_);
            for (auto box : bbox_per_class[i]) {
                box.x *= output_width_scaling_;
                box.y *= output_height_scaling_;
                box.w *= output_width_scaling_;
                box.h *= output_height_scaling_;
                *inserter = box;
                ++inserter;
            }
        }
    }

    /// @brief Sets Intersection Over Union (IOU) threshold
    /// @param threshold Threshold on IOU metrics to consider that two boxes are matching
    /// @note Intersection Over Union (IOU) is the ratio of the intersection area over union area
    void set_iou_threshold(float threshold) {
        iou_threshold_ = threshold;
    }

    /// @brief Configures the computation to ignore some class identifier
    /// @param class_id Identifier of the class to be ignored
    void ignore_class_id(std::size_t class_id) {
        assert(valid_classes_.size() == num_classes_);
        if (class_id >= num_classes_) {
            std::ostringstream oss;
            oss << "ERROR: bad class_id: " << class_id << ". num_classes: " << num_classes_ << std::endl;
            throw std::invalid_argument(oss.str());
        }
        valid_classes_[class_id] = false;
    }

    /// @brief Suppresses non-maximum overlapping boxes over a list of EventBbox-es.
    /// @note The list is modified in-place. The result is sorted by confidence.
    /// @param[in,out] bbox_list List of @ref EventBbox on which to apply the Non-maximum suppression
    /// @param iou_threshold Threshold above which two boxes are considered to overlap
    static void compute_nms_per_class(std::list<EventBbox> &bbox_list, float iou_threshold) {
        if (bbox_list.empty()) {
            return;
        }

        // Sort by confidence.
        bbox_list.sort([](const EventBbox &a, const EventBbox &b) { return a.class_confidence > b.class_confidence; });

        // Loop until we process all elements:
        // max_bbox_it is the best box we are considering each time.
        for (auto max_bbox_it = bbox_list.begin(); max_bbox_it != bbox_list.end(); ++max_bbox_it) {
            // Check the IoU with all other boxes.
            for (auto bbox_it = std::next(max_bbox_it); bbox_it != bbox_list.end(); /**/) {
                float iou = max_bbox_it->intersection_area_over_union(*bbox_it);
                if (iou > iou_threshold) {
                    bbox_list.erase(bbox_it++);
                } else {
                    ++bbox_it;
                }
            }
        }
    }

private:
    std::size_t num_classes_     = 0;
    int events_input_width_      = 0;
    int events_input_height_     = 0;
    int network_input_width_     = 0;
    int network_input_height_    = 0;
    float output_width_scaling_  = 1.f;
    float output_height_scaling_ = 1.f;
    float iou_threshold_         = 0.f;
    std::vector<bool> valid_classes_;
};

} // namespace Metavision

#endif // METAVISION_SDK_ML_ALGORITHMS_NON_MAXIMUM_SUPPRESSION_H
