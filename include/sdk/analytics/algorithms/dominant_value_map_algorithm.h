/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_DOMINANT_VALUE_MAP_ALGORITHM_H
#define METAVISION_SDK_ANALYTICS_DOMINANT_VALUE_MAP_ALGORITHM_H

#include <stdexcept>
#include <opencv2/opencv.hpp>

#include "metavision/sdk/analytics/utils/histogram_utils.h"

namespace Metavision {

/// @brief Class computing the dominant value of a map
/// @tparam precision_type Return type of the dominant value, used to define the type of the boundaries of the histogram
/// bins
template<typename precision_type = float>
class DominantValueMapAlgorithm {
public:
    /// @brief Constructor
    ///
    /// We split the range [ @p min_val, @p max_val ] to get values spaced apart by @p precision_val .
    /// Bins are centered around these values and are of width step so that consecutive bins touch each other.
    /// For example, given the range [3, 5] and @p precision_val = 1, it will compute the bin centers {3, 4, 5},
    /// the boundaries of which are given by {2.5, 3.5, 4.5, 5.5}
    ///
    /// @param min_val Minimum included value (lower bound of the histogram bins)
    /// @param max_val Maximum included value (upper bound of the histogram bins)
    /// @param precision_val Width of the bins of the histogram (same unit as the value to estimate)
    /// @param min_count Minimum size of a given bin in the histogram to be eligible as dominant
    /// @note The histogram bins are initialized during the class construction and won't change dynamically.
    DominantValueMapAlgorithm(precision_type min_val, precision_type max_val, precision_type precision_val,
                              unsigned int min_count);

    /// @brief Destructor
    ~DominantValueMapAlgorithm() = default;

    /// @brief Computes the dominant value from a value map
    /// @param value_map Input map containing values
    /// @param output_dominant_value Peak of the histogram of the values contained in @p value_map
    /// @return false if none of the bins were sufficiently filled with respect to the count criterion
    bool compute_dominant_value(const cv::Mat &value_map, precision_type &output_dominant_value);

private:
    /// @brief Computes the dominant value from a map containing the values
    /// @param value_map Input map containing values of type @p precision_type
    /// @param output_peak_id ID of the histogram's peak
    /// @return false if none of the bins were sufficiently filled with respect to the count criterion
    bool calculate_histogram_and_find_peak(const cv::Mat_<precision_type> &value_map, int &output_peak_id);

    unsigned int min_count_; ///< Minimum size of a given bin in the histogram to be eligible as dominant

    std::vector<precision_type> hist_bins_boundaries_;
    std::vector<precision_type> hist_bins_centers_;
    cv::Mat_<precision_type> precision_value_map_; ///< Auxiliary image to store the values using the precision_type
    cv::Mat_<precision_type> histogram_;           ///< Auxiliary image to store the histogram
};

template<typename precision_type>
DominantValueMapAlgorithm<precision_type>::DominantValueMapAlgorithm(precision_type min_val, precision_type max_val,
                                                                     precision_type precision_val,
                                                                     unsigned int min_count) :
    min_count_(min_count) {
    Metavision::init_histogram_bins<precision_type>(min_val, max_val, precision_val, hist_bins_centers_,
                                                    hist_bins_boundaries_);
}

template<typename precision_type>
bool DominantValueMapAlgorithm<precision_type>::compute_dominant_value(const cv::Mat &value_map,
                                                                       precision_type &output_dominant_value) {
    int output_peak_id;
    // Convert cv::Mat if necessary
    if (value_map.depth() == precision_value_map_.depth()) {
        if (!calculate_histogram_and_find_peak(value_map, output_peak_id))
            return false;
    } else {
        precision_value_map_.create(value_map.size());
        value_map.convertTo(precision_value_map_, precision_value_map_.depth());
        if (!calculate_histogram_and_find_peak(precision_value_map_, output_peak_id))
            return false;
    }
    output_dominant_value = hist_bins_centers_[output_peak_id];
    return true;
}

template<typename precision_type>
bool DominantValueMapAlgorithm<precision_type>::calculate_histogram_and_find_peak(
    const cv::Mat_<precision_type> &value_map, int &output_peak_id) {
    // Compute the histogram
    const float *histRange = hist_bins_boundaries_.data();
    const int nbins        = static_cast<int>(hist_bins_boundaries_.size() - 1);
    cv::calcHist(&value_map, 1, 0, cv::Mat(), histogram_, 1, &nbins, &histRange, false, false);

    // Compute and return the ID corresponding to the peak
    double max_bin_count;
    cv::Point max_bin_loc;
    cv::minMaxLoc(histogram_, 0, &max_bin_count, 0, &max_bin_loc);
    if (max_bin_count < min_count_)
        return false;
    output_peak_id = max_bin_loc.y;
    return (output_peak_id >= 0 && output_peak_id < static_cast<int>(hist_bins_centers_.size()));
}

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_DOMINANT_VALUE_MAP_ALGORITHM_H
