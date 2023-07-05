/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_EVENT_TRACKING_DATA_H
#define METAVISION_SDK_ANALYTICS_EVENT_TRACKING_DATA_H

#include <cmath>
#include <limits>

#include "metavision/sdk/base/events/event2d.h"

namespace Metavision {

/// @brief Structure representing a tracked object that is produced by the @ref TrackingAlgorithm. More precisely, this
/// structure  represents the bounding box around the events generated by the tracked object during a given time slice.
struct EventTrackingData {
    /// @brief Default constructor.
    EventTrackingData() = default;

    /// @brief Constructor.
    /// @param x Bounding box's center's abscissa coordinate.
    /// @param y Bounding box's center's ordinate coordinate.
    /// @param t Current track's timestamp.
    /// @param width Bounding box's width.
    /// @param height Bounding box's height.
    /// @param object_id Tracked object's ID.
    EventTrackingData(double x, double y, timestamp t, double width, double height, size_t object_id);

    /// @brief Constructor.
    /// @param x Bounding box's center's abscissa coordinate.
    /// @param y Bounding box's center's ordinate coordinate.
    /// @param t Current track's timestamp.
    /// @param width Bounding box's width.
    /// @param height Bounding box's height.
    /// @param object_id Tracked object's ID.
    /// @param event_id Instance's ID.
    EventTrackingData(double x, double y, timestamp t, double width, double height, size_t object_id, size_t event_id);

    /// @brief Comparison operator.
    /// @param ev The event to compare with the current instance.
    /// @return true if the two events are equal.
    bool operator==(const EventTrackingData &ev) const {
        return std::abs(x_ - ev.x_) < std::numeric_limits<double>::epsilon() &&
               std::abs(y_ - ev.y_) < std::numeric_limits<double>::epsilon() &&
               std::abs(width_ - ev.width_) < std::numeric_limits<double>::epsilon() &&
               std::abs(height_ - ev.height_) < std::numeric_limits<double>::epsilon() && object_id_ == ev.object_id_;
    }

    /// @brief Comparison operator.
    /// @param ev The event to compare with the current instance.
    /// @return true if the two events are different.
    bool operator!=(const EventTrackingData &ev) const {
        return !(*this == ev);
    }

    /// @brief Column position in the sensor at which the event happened
    unsigned short x;

    /// @brief Row position in the sensor at which the event happened
    unsigned short y;

    /// @brief Timestamp at which the event happened (in us)
    timestamp t;

    double x_, y_;          ///< Bounding box's center's coordinates.
    double width_, height_; ///< Bounding box's dimensions.
    size_t object_id_;      ///< Tracked object's ID.
    size_t event_id_;       ///< Instance's ID.

    /// @brief Deserializes an EventTrackingData from a buffer.
    /// @param buf Pointer to the raw buffer from which the event has to be deserialized.
    /// @param delta_ts Time delta to add to the deserialized event's timestamp.
    /// @return The deserialized event.
    static EventTrackingData read_event(void *buf, const timestamp &delta_ts = 0) {
        RawEvent *buffer = static_cast<RawEvent *>(buf);
        return EventTrackingData(buffer->x_, buffer->y_, buffer->ts + delta_ts, buffer->width_, buffer->height_,
                                 buffer->object_id_, buffer->event_id_);
    }

    /// @brief Serializes an EventTrackingData in a buffer.
    /// @param buf Pointer to the raw buffer in which the event has to be serialized.
    /// @param origin Time delta to subtract from the event's timestamp.
    void write_event(void *buf, timestamp origin) const {
        RawEvent *buffer   = static_cast<RawEvent *>(buf);
        buffer->ts         = static_cast<uint32_t>(t - origin);
        buffer->x          = x;
        buffer->y          = y;
        buffer->x_         = x_;
        buffer->y_         = y_;
        buffer->width_     = width_;
        buffer->height_    = height_;
        buffer->object_id_ = object_id_;
        buffer->event_id_  = event_id_;
    }

    FORCE_PACK(
        /// @brief Structure of size 96 bits to represent one event.
        struct RawEvent {
            uint32_t ts;
            unsigned int x : 14;
            unsigned int y : 14;
            double x_;
            double y_;
            double width_;
            double height_;
            size_t object_id_;
            size_t event_id_;
        });

private:
    static size_t g_event_id;
};

} // namespace Metavision

METAVISION_DEFINE_EVENT_TRAIT(Metavision::EventTrackingData, 100, "Tracking")

#endif // METAVISION_SDK_ANALYTICS_EVENT_TRACKING_DATA_H
