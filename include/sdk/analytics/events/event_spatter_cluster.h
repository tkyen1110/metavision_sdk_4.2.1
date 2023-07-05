/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_EVENT_SPATTER_CLUSTER_H
#define METAVISION_SDK_ANALYTICS_EVENT_SPATTER_CLUSTER_H

#include <opencv2/opencv.hpp>

#include "metavision/sdk/base/events/event2d.h"
#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Structure representing a cluster of events
struct EventSpatterCluster {
    /// x coordinate of the upper left corner of the cluster's bounding box
    float x;
    /// y coordinate of the upper left corner of the cluster's bounding box
    float y;
    /// Width of the cluster's bounding box
    float width;
    /// Height of the cluster's bounding box
    float height;

    /// Cluster's ID
    int id = -1;

    /// Number of untracked times
    int untracked_times = 0;

    /// Timestamp of the detection
    timestamp t;

    /// @brief Returns the center of the cluster
    /// @return The center of the cluster
    const cv::Point2f get_centroid() const {
        cv::Point2f centroid;
        centroid.x = x + 0.5f * width;
        centroid.y = y + 0.5f * height;
        return centroid;
    }

    /// @brief Returns the rectangle defined by the cluster
    /// @return Rectangle defined by the cluster
    const cv::Rect2f get_rect() const {
        return cv::Rect2f(x, y, width, height);
    }

    /// @brief Default constructor
    EventSpatterCluster() = default;

    /// @brief Constructor
    /// @param x X coordinate of the rectangle defined by the cluster
    /// @param y Y coordinate of the rectangle defined by the cluster
    /// @param width Width of the rectangle defined by the cluster
    /// @param height Height of the rectangle defined by the cluster
    /// @param id Cluster ID
    /// @param untracked_times Number of untracked times
    /// @param ts Timestamp of the detection
    inline EventSpatterCluster(float x, float y, float width, float height, int id, int untracked_times, timestamp ts) :
        x(x), y(y), width(width), height(height), id(id), untracked_times(untracked_times), t(ts) {}

    /// @brief Constructor
    /// @param x X coordinate of the rectangle defined by the cluster
    /// @param y Y coordinate of the rectangle defined by the cluster
    /// @param width Width of the rectangle defined by the cluster
    /// @param height Height of the rectangle defined by the cluster
    /// @param ts Timestamp of the detection
    inline EventSpatterCluster(float x, float y, float width, float height, timestamp ts) :
        EventSpatterCluster(x, y, width, height, -1, 0, ts) {}

    /// @brief Writes EventSpatterCluster to buffer
    void write_event(void *buf, timestamp origin) const {
        RawEvent *buffer        = (RawEvent *)buf;
        buffer->ts              = t - origin;
        buffer->x               = x;
        buffer->y               = y;
        buffer->width           = width;
        buffer->height          = height;
        buffer->id              = id;
        buffer->untracked_times = untracked_times;
    }

    /// @brief Reads event 2D from buffer
    /// @return Event spatter cluster
    static EventSpatterCluster read_event(void *buf, const timestamp &delta_ts) {
        RawEvent *buffer = static_cast<RawEvent *>(buf);
        return EventSpatterCluster(buffer->x, buffer->y, buffer->width, buffer->height, buffer->id,
                                   buffer->untracked_times, buffer->ts + delta_ts);
    }

    /// @brief Returns the size of the RawEvent
    /// @return The size of the RawEvent
    static size_t get_raw_event_size() {
        return sizeof(RawEvent);
    }

    /// @brief Operator <<
    friend std::ostream &operator<<(std::ostream &output, const EventSpatterCluster &e) {
        output << "EventSpatterCluster: (";
        output << e.x << ", " << e.y << ", " << e.width << ", " << e.height << ", " << e.t << ", id: " << e.id
               << ", untracked times: " << e.untracked_times << ", ";
        output << ")";
        return output;
    }

    FORCE_PACK(
        /// @brief Structure representing one event
        struct RawEvent {
            unsigned int ts : 32;
            float x;
            float y;
            float width;
            float height;
            int id;
            int untracked_times;
        });
};
} // namespace Metavision

METAVISION_DEFINE_EVENT_TRAIT(Metavision::EventSpatterCluster, 101, "EventSpatterCluster")

#endif // METAVISION_SDK_ANALYTICS_EVENT_SPATTER_CLUSTER_H
