/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_EVENT_OPTICAL_FLOW_H
#define METAVISION_SDK_CV_EVENT_OPTICAL_FLOW_H

#include "metavision/sdk/base/utils/detail/struct_pack.h"
#include "metavision/sdk/base/events/detail/event_traits.h"
#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Class representing an event used to describe an optical flow
struct EventOpticalFlow {
public:
    /// @brief Default constructor
    EventOpticalFlow() = default;

    /// @brief Constructor
    /// @param x Column position of the event in the sensor
    /// @param y Row position of the event in the sensor
    /// @param p Polarity specialising the event
    /// @param t Timestamp of the event (in us)
    /// @param vx Speed in X axis in pixels per second in the image plane of the considered event
    /// @param vy Speed in Y axis in pixels per second in the image plane of the considered event
    /// @param id Feature ID of the considered event
    /// @param cx Horizontal coordinate of the center of the feature used to compute speed of the considered event
    /// @param cy Vertical coordinate of the center of the feature used to compute speed of the considered event
    EventOpticalFlow(unsigned short x, unsigned short y, short p, timestamp t, float vx, float vy, unsigned int id,
                     float cx, float cy) :
        x(x), y(y), p(p), t(t), vx(vx), vy(vy), id(id), center_x(cx), center_y(cy) {}

    /// @brief Column position in the sensor at which the event happened
    unsigned short x;

    /// @brief Row position in the sensor at which the event happened
    unsigned short y;

    /// @brief Polarity, whose value depends on the type of the event (CD or EM)
    /// @sa @ref Metavision::Event2d for more details
    short p;

    /// @brief Timestamp at which the event happened (in us)
    timestamp t;

    /// @brief Speed in X axis in pixels per second in the image plane of the considered event
    float vx;

    /// @brief Speed in Y axis in pixels per second in the image plane of the considered event
    float vy;

    /// @brief id feature ID of the considered event.
    unsigned int id = 0;

    /// @brief center_x horizontal coordinate of the center of the feature used to compute speed of the considered
    /// event.
    float center_x = 0.f;

    /// @brief center_y vertical coordinate of the center of the feature used to compute speed of the considered event.
    float center_y = 0.f;

    /// @brief Read EventOpticalFlowCCL from buffer
    inline static EventOpticalFlow read_event(void *buf, const timestamp &delta_ts);

    /// @brief Write EventOpticalFlowCCL in buffer
    inline void write_event(void *buf, timestamp origin) const;

    FORCE_PACK(
        /// RAW event format of a EventOpticalFlowCCL event
        struct RawEvent {
            uint32_t ts;
            unsigned int x : 14;
            unsigned int y : 14;
            unsigned int p : 4;
            float vx;
            float vy;
            float center_x;
            float center_y;
            uint32_t id;
        });
};
} // namespace Metavision

#include "detail/event_optical_flow_impl.h"

METAVISION_DEFINE_EVENT_TRAIT(Metavision::EventOpticalFlow, 40, "EventOpticalFlow")

#endif // METAVISION_SDK_CV_EVENT_OPTICAL_FLOW_H
