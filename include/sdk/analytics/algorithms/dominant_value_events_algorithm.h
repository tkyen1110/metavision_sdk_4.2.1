/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_DOMINANT_VALUE_EVENTS_ALGORITHM_H
#define METAVISION_SDK_ANALYTICS_DOMINANT_VALUE_EVENTS_ALGORITHM_H

#include <stdexcept>
#include <opencv2/opencv.hpp>

#include "metavision/sdk/analytics/utils/histogram_utils.h"

namespace Metavision {

/// @brief Class computing the dominant value of an event's field from an events batch
/// @tparam T Type for which the dominant value of one of its fields will be estimated
/// @tparam V Type of the field from which the dominant value is extracted
/// @tparam v Address in @p T of the field from which the dominant value is extracted
/// @tparam precision_type Return type of the dominant value, used to define the type of the boundaries of the histogram
/// bins
template<typename T, typename V, V T::*v, typename precision_type = float>
class DominantValueEventsAlgorithm {
public:
    /// @brief Constructor
    ///
    /// We split the range [ @p min_val, @p max_val ] to get values spaced by @p precision_val .
    /// Bins are centered around these values and are of a width that ensures that consecutive bins touch each other.
    /// For example, given the range [3, 5] and @p precision_val = 1, it will compute the bin centers {3, 4, 5},
    /// the boundaries of which are given by {2.5, 3.5, 4.5, 5.5}
    ///
    /// @note The histogram bins are initialized during the class construction and won't change dynamically.
    /// @param min_val Minimum included value (lower bound of the histogram bins)
    /// @param max_val Maximum included value (upper bound of the histogram bins)
    /// @param precision_val Width of the bins of the histogram (same unit as the value to estimate)
    /// @param min_count Minimum size of a given bin in the histogram to be eligible as dominant
    DominantValueEventsAlgorithm(precision_type min_val, precision_type max_val, precision_type precision_val,
                                 unsigned int min_count) :
        min_count_(min_count) {
        Metavision::init_histogram_bins<precision_type>(min_val, max_val, precision_val, hist_bins_centers_,
                                                        hist_bins_boundaries_);
        assert(hist_bins_centers_.size() != 0);
        counts_.resize(hist_bins_centers_.size(), 0);
    }

    /// @brief Computes the dominant value from a batch of events
    /// @tparam InputIt An iterator type over an event of type @p T
    /// @param begin First iterator of the @p T events buffer to process
    /// @param end End iterator of the @p T events buffer to process
    /// @param output_dominant_value Peak of the histogram of the values contained between @p begin and @p end
    /// @return false if none of the bins were sufficiently filled with respect to the count criterion
    template<typename InputIt>
    bool compute_dominant_value(InputIt begin, InputIt end, precision_type &output_dominant_value) {
        assert(counts_.size() == hist_bins_centers_.size());
        std::fill(counts_.begin(), counts_.end(), 0); // Reset counts

        for (auto it = begin; it != end; it++) {
            const precision_type ev_val = static_cast<precision_type>((*it).*v);
            size_t id_bin;
            if (Metavision::value_to_histogram_bin_id(hist_bins_boundaries_, ev_val, id_bin))
                counts_[id_bin]++;
        }

        // Find the peak
        const auto it_peak = std::max_element(counts_.cbegin(), counts_.cend());
        if (*it_peak < min_count_)
            return false;
        const auto id_peak = std::distance(counts_.cbegin(), it_peak);
        assert(id_peak >= 0 && id_peak < hist_bins_centers_.size());
        output_dominant_value = hist_bins_centers_[id_peak];
        return true;
    }

private:
    unsigned int min_count_; ///< Minimum size of a given bin in the histogram to be eligible as dominant

    std::vector<precision_type> hist_bins_boundaries_;
    std::vector<precision_type> hist_bins_centers_;
    std::vector<unsigned int> counts_;
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_DOMINANT_VALUE_EVENTS_ALGORITHM_H
