/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_MONO_COUNTING_STATUS_H
#define METAVISION_SDK_ANALYTICS_MONO_COUNTING_STATUS_H

#include <map>
#include <opencv2/core/types.hpp>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Structure to store the mono counting results on several lines
struct MonoCountingStatus {
    MonoCountingStatus() {
        reset();
    }

    void reset() {
        global_counter = 0;
        line_mono_counters.clear();
    }

    bool operator==(const MonoCountingStatus &o) const {
        return (global_counter == o.global_counter && line_mono_counters == o.line_mono_counters);
    }

    bool operator!=(const MonoCountingStatus &o) const {
        return !(*this == o);
    }
    int global_counter;                    ///< Maximum count over the line counters
    std::map<int, int> line_mono_counters; ///< Row and counts of each line counter

    timestamp last_count_ts; ///< Timestamp of the last triggered count
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_MONO_COUNTING_STATUS_H
