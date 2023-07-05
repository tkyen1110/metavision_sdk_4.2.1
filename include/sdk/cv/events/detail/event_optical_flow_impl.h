/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_DETAIL_EVENT_OPTICAL_FLOW_IMPL_H
#define METAVISION_SDK_CV_DETAIL_EVENT_OPTICAL_FLOW_IMPL_H

namespace Metavision {

EventOpticalFlow EventOpticalFlow::read_event(void *buf, const timestamp &delta_ts) {
    EventOpticalFlow::RawEvent *buffer = static_cast<EventOpticalFlow::RawEvent *>(buf);
    EventOpticalFlow ev;
    ev.t        = buffer->ts + delta_ts;
    ev.x        = buffer->x;
    ev.y        = buffer->y;
    ev.p        = buffer->p;
    ev.vx       = buffer->vx;
    ev.vy       = buffer->vy;
    ev.id       = buffer->id;
    ev.center_x = buffer->center_x;
    ev.center_y = buffer->center_y;
    return ev;
}

void EventOpticalFlow::write_event(void *buf, timestamp origin) const {
    EventOpticalFlow::RawEvent *buffer = (EventOpticalFlow::RawEvent *)buf;
    buffer->ts                         = static_cast<uint32_t>(t - origin);
    buffer->x                          = x;
    buffer->y                          = y;
    buffer->p                          = p;
    buffer->id                         = id;
    buffer->vx                         = vx;
    buffer->vy                         = vy;
    buffer->center_x                   = center_x;
    buffer->center_y                   = center_y;
}

} // namespace Metavision

#endif // METAVISION_SDK_CV_DETAIL_EVENT_OPTICAL_FLOW_IMPL_H
