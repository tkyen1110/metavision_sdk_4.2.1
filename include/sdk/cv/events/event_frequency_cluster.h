/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_EVENT_FREQUENCY_CLUSTER_H
#define METAVISION_SDK_CV_EVENT_FREQUENCY_CLUSTER_H

#include <type_traits>
#include <cstdint>

#include "metavision/sdk/base/events/detail/event_traits.h"
#include "metavision/sdk/base/utils/detail/struct_pack.h"
#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Event2dFrequencyCluster represents a cluster of frequency events.
template<typename T = float>
struct Event2dFrequencyCluster {
    static_assert(std::is_floating_point<T>::value,
                  "Event2dFrequencyCluster can only be instantiated with a floating point type.");

public:
    T x               = 0; ///< X coordinate of the center of the cluster.
    T y               = 0; ///< Y coordinate of the center of the cluster.
    timestamp t       = 0; ///< Timestamp of the last update (in us).
    T frequency       = 0; ///< Filtered frequency of events associated to the cluster.
    uint32_t id       = 0; ///< Cluster ID.
    uint32_t n_pixels = 0; ///< Number of different pixels "triggered" by an event, belonging to the cluster.
    uint32_t n_events = 0; ///< Number of events that were associated to this cluster.

    Event2dFrequencyCluster() = default;

    Event2dFrequencyCluster(T x, T y, timestamp t, T frequency, uint32_t id) :
        x(x), y(y), t(t), frequency(frequency), id(id), n_pixels(0), n_events(0) {}

    ~Event2dFrequencyCluster() = default;

    void write_event(void *buf, timestamp origin) const {
        RawEvent *buffer  = static_cast<RawEvent *>(buf);
        buffer->ts        = t - origin;
        buffer->x         = x;
        buffer->y         = y;
        buffer->frequency = frequency;
        buffer->id        = id;
    }

    FORCE_PACK(struct RawEvent {
        uint32_t ts;
        unsigned int x : 14;
        unsigned int y : 14;
        uint32_t id;
        float frequency;
    });
};

} // namespace Metavision

METAVISION_DEFINE_EVENT_TRAIT(Metavision::Event2dFrequencyCluster<double>, 110, "Frequency cluster double")
METAVISION_DEFINE_EVENT_TRAIT(Metavision::Event2dFrequencyCluster<float>, 111, "Frequency cluster float")

#endif // METAVISION_SDK_CV_EVENT_FREQUENCY_CLUSTER_H
