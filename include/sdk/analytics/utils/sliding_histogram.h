/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_SLIDING_HISTOGRAM_H
#define METAVISION_SDK_ANALYTICS_SLIDING_HISTOGRAM_H

#include <assert.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>

#include "metavision/sdk/analytics/utils/histogram_utils.h"
#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

class SlidingHistogram {
public:
    using OutputCb = std::function<void(timestamp, const std::vector<unsigned int> &)>;

    using precision_type = float;

    /// @brief Constructor
    ///
    /// We split the range [ @p min_val, @p max_val ] to get values spaced apart by @p precision_val .
    /// Bins are centered around these values and are of width @p precision_val so that consecutive bins touch each
    /// other. For example, given the range [3, 5] and @p precision_val = 1, it will compute the bin centers {3, 4, 5},
    /// the boundaries of which are given by {2.5, 3.5, 4.5, 5.5}
    ///
    /// @note The histogram bins are initialized during the class construction and won't change dynamically.
    /// @param min_val Minimum included value (lower bound of the histogram bins)
    /// @param max_val Maximum included value (upper bound of the histogram bins)
    /// @param precision_val Width of the bins of the histogram (same unit as the value to estimate)
    /// @param accumulation_time_us Accumulation time of the histogram (in us), a negative value disables this option
    /// and old values are not discarded
    /// @param output_period_us Period (in us) between two histograms generations. The period is measured with the
    /// input events' timestamp
    SlidingHistogram(precision_type min_val, precision_type max_val, precision_type precision_val,
                     timestamp accumulation_time_us, timestamp output_period_us);

    /// @brief Adds a new timestamped measure
    /// @param ts Timestamp of the measure
    /// @param val Measured value
    /// @return false if the value is outside the extreme values of the bins
    template<typename T>
    bool add_time_and_value(timestamp ts, T val);

    /// @brief Specifies the current processing timestamp to the class, to let it know that there's no new measure up to
    /// this timestamp
    /// @param ts Current processing timestamp
    void add_time(timestamp ts);

    /// @brief Function to pass a callback to know when a histogram is available
    inline void set_output_callback(const OutputCb &output_cb) {
        output_cb_ = output_cb;
    }

    inline const std::vector<precision_type> &get_histogram_centers() {
        return hist_bins_centers_;
    }

private:
    struct TimeId {
        TimeId(timestamp t, size_t id) : t(t), id(id) {}
        timestamp t;
        size_t id;
    };

    /// @brief Removes from the histogram measures older than the accumulation time, and outputs the new histogram via
    /// the callback
    /// @param produce_ts Timestamp at which to produce the histogram
    void produce_histogram(timestamp produce_ts);

    OutputCb output_cb_;

    std::vector<precision_type> hist_bins_boundaries_;
    std::vector<precision_type> hist_bins_centers_;
    std::vector<unsigned int> counts_;

    std::vector<TimeId> timestamped_ids_;
    timestamp ts_offset_; ///< State variable to handle time overflow in the events history. This is to minimize the
                          ///< memory footprint of the events history access.

    timestamp accumulation_time_us_;
    timestamp next_produce_time_us_;
    timestamp output_period_us_; ///< Period (in us) between two histogram generations (i.e. period between two calls to
                                 ///< asynchronous processing calls). The period is measured with the input events'
                                 ///< timestamp.
};

template<typename T>
bool SlidingHistogram::add_time_and_value(timestamp ts, T val) {
    add_time(ts);

    size_t id_bin;
    if (!Metavision::value_to_histogram_bin_id(hist_bins_boundaries_, val, id_bin))
        return false;

    timestamped_ids_.emplace_back(ts - ts_offset_, id_bin);
    counts_[id_bin]++;
    return true;
}

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_SLIDING_HISTOGRAM_H