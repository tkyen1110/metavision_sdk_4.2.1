/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_HISTOGRAM_UTILS_H
#define METAVISION_SDK_ANALYTICS_HISTOGRAM_UTILS_H

#include "metavision/sdk/base/utils/sdk_log.h"

namespace Metavision {

/// @brief Computes the histogram bins given a minimum value, a maximum value and a step value.
///
/// We split the range [ @p min_val, @p max_val ] to get values spaced apart by @p step .
/// Bins are centered around these values and are of width @p step so that consecutive bins touch each other.
/// For example, given the range [3, 5] and @p step = 1, it will compute the bin centers {3, 4, 5},
/// the boundaries of which are given by {2.5, 3.5, 4.5, 5.5}
///
/// @tparam T Type used to define the type of the boundaries of the histogram bins
/// @param min_val Minimum included value (lower bound of the histogram bins)
/// @param max_val Maximum included value (upper bound of the histogram bins)
/// @param step Width of the bins of the histogram
/// @param bins_centers Output vector containing the bin centers, e.g. {3, 4, 5}
/// @param bins_boundaries Output vector containing the bin boundaries, e.g. {2.5, 3.5, 4.5, 5.5}
/// @return false if it fails
template<typename T>
bool init_histogram_bins(T min_val, T max_val, T step, std::vector<T> &bins_centers, std::vector<T> &bins_boundaries) {
    static_assert(std::is_floating_point<T>::value, "Float type required");
    if (min_val >= max_val) {
        MV_SDK_LOG_ERROR() << "The minimum value must be strictly smaller than the maximum value.";
        return false;
    }
    if (step <= 0) {
        MV_SDK_LOG_ERROR() << "The width of the histogram bins must be strictly positive.";
        return false;
    }

    const T range_val = max_val - min_val;
    if (step > range_val) {
        MV_SDK_LOG_ERROR() << "The range of the values must be greater than the width of the histogram bins.";
        return false;
    }

    bins_centers.clear();
    bins_boundaries.clear();

    // We split the range [min_val, max_val] to get values spaced apart by step
    const int num_bins_centers = 1 + static_cast<int>(range_val / step + T(0.5));
    bins_centers.reserve(num_bins_centers);
    // Bins are centered around these values and are of width step so that consecutive bins touch each other
    bins_boundaries.reserve(num_bins_centers + 1);

    T center     = min_val;
    T left_bound = min_val - T(0.5) * step;
    for (; left_bound < max_val; center += step, left_bound += step) {
        bins_centers.emplace_back(center);
        bins_boundaries.emplace_back(left_bound);
    }
    bins_boundaries.emplace_back(left_bound); // Last bin (upper bound)
    assert(bins_centers.size() == num_bins_centers);
    return true;
}

/// @brief Finds the ID of the histogram bin that corresponds to the input value
/// @param bins_boundaries Vector containing the histogram bin boundaries, e.g. {2.5, 3.5, 4.5, 5.5} when bin centers
/// are {3, 4, 5}
/// @param value Input value that is compared against the bins values
/// @param output_id ID of the bin that matches the input value. N.B. This is the ID of the center, not of the boundary.
/// @return false if the value is outside the bins range
template<typename T>
bool value_to_histogram_bin_id(const std::vector<T> &bins_boundaries, T value, size_t &output_id) {
    assert(!bins_boundaries.empty());
    if (value < bins_boundaries.front() || value > bins_boundaries.back())
        return false;
    // Centers ID:       0       1       2       3       4       5
    //               |   x   |   x   |   x   |   x   |   x   |   x   |
    //   Bound ID:   0       1       2       3       4       5       6
    //
    // If the value falls near the center ID 1, then 2 is the first bound that is greater than the value.
    const auto it_bin = std::lower_bound(bins_boundaries.cbegin(), bins_boundaries.cend(), value,
                                         [](const auto &left__bound, const T &val) { return left__bound < val; });

    output_id = std::distance(bins_boundaries.cbegin(), it_bin) - 1;
    return true;
}

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_HISTOGRAM_UTILS_H