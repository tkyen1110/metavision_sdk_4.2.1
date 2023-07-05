/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_EVENT_PROVIDER_H
#define METAVISION_SDK_ML_EVENT_PROVIDER_H

#include <string>

#include "metavision/sdk/core/algorithms/file_producer_algorithm.h"
#include "metavision/sdk/driver/camera.h"

namespace Metavision {

/// @brief Wraps @ref Metavision::Device to generate a generic event producer
class EventProviderBase {
public:
    using EventCallback = std::function<void(const EventCD *, const EventCD *)>;

    virtual ~EventProviderBase() {}

    /// @brief Configures the device callback
    /// @param cb Function to be called on received events
    void set_callback(EventCallback cb) {
        cb_ = cb;
    }

    /// @brief Returns Sensor's width
    /// @return Sensor's width
    virtual int get_width() = 0;

    /// @brief Returns Sensor's height
    /// @return Sensor's height
    virtual int get_height() = 0;

    /// @brief Starts streaming events
    virtual void start() = 0;

    /// @brief Sets event rate limit
    /// @return true on success
    virtual bool set_event_rate_limit(uint32_t ev_rate) {
        return false;
    }

    /// @brief Stops streaming events
    void stop() {
        done_ = true;
    }

    /// @brief Checks if the camera is stopped
    /// @return true if the camera is stopped, False otherwise
    bool is_done() {
        return done_;
    }

    /// @brief Sets after which the callback should be called
    /// @param start_ts Timestamp of the first useful event
    void set_start_ts(timestamp start_ts) {
        start_ts_ = start_ts;
    }

    /// @brief Sets the timestamp of the last considered event
    /// @param end_ts Timestamp of the last useful event
    void set_end_ts(timestamp end_ts) {
        end_ts_ = end_ts;
    }

    /// @brief Gets the first considered timestamp
    /// @return first considered timestamp
    timestamp get_start_ts() const {
        return start_ts_;
    }

protected:
    EventCallback cb_   = nullptr;
    bool done_          = false;
    timestamp start_ts_ = 0;
    timestamp end_ts_   = std::numeric_limits<timestamp>::max();
};

/// @brief Implementing EventProviderBase abstraction for RAW files and physical devices
class EventProviderRaw : public EventProviderBase {
public:
    /// @brief Creates a virtual camera
    /// @param filename If empty (default value), opens the first available camera. Otherwise loads the corresponding
    /// RAW file
    EventProviderRaw(const std::string filename = "");

    virtual ~EventProviderRaw() {}

    /// @brief Returns the sensor's width
    /// @return Sensor's width
    virtual int get_width() {
        return camera_.geometry().width();
    }

    /// @brief Returns the sensor's height
    /// @return Sensor's height
    virtual int get_height() {
        return camera_.geometry().height();
    }

    /// @brief Starts the camera / the processing
    virtual void start();

    /// @brief Sets event rate limit
    /// @return true on success
    virtual bool set_event_rate_limit(uint32_t ev_rate) {
        try {
            camera_.erc_module().set_cd_event_rate(ev_rate);
            camera_.erc_module().enable(true);
            return true;
        } catch (Metavision::CameraException &e) {}

        return false;
    }

private:
    /// @brief Ignores events before start_ts and stop when we reach end_ts
    /// @param ev_begin pointer on the first event
    /// @param ev_end pointer on the last event
    void filter_ts(const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end);
    Camera camera_;
};

/// @brief Implementing EventProviderBase abstraction for DAT files
class EventProviderDat : public EventProviderBase {
public:
    /// @brief Constructs an event provider able to read DAT files
    /// @param filename Input DAT file name
    EventProviderDat(const std::string filename) : prod_(new FileProducerAlgorithm(filename)){};

    virtual ~EventProviderDat() {}

    /// @brief Returns the sensor's width
    /// @return Sensor's width
    virtual int get_width() {
        return prod_->get_width();
    }

    /// @brief Returns the sensor's height
    /// @return Sensor's height
    virtual int get_height() {
        return prod_->get_height();
    }

    /// @brief Starts streaming events
    virtual void start();

private:
    std::unique_ptr<FileProducerAlgorithm> prod_;
    const timestamp chunk_duration_ = 1000; ///< 1 ms
    timestamp current_ts_           = 0;
};

/// @brief Creates a virtual camera
/// @param filename If empty (default value), opens the first available camera. Otherwise loads the corresponding
/// RAW file
EventProviderRaw::EventProviderRaw(const std::string filename) {
    if (filename == "") {
        // live stream from camera
        camera_ = Metavision::Camera::from_first_available();
    } else {
        // from existing file
        camera_ = Camera::from_file(filename, FileConfigHints().real_time_playback(false));
    }
    camera_.cd().add_callback(
        std::bind(&EventProviderRaw::filter_ts, this, std::placeholders::_1, std::placeholders::_2));
}

/// @brief Starts the camera / the processing
void EventProviderRaw::start() {
    done_ = false;
    if (cb_ == nullptr) {
        done_ = true;
    } else {
        camera_.start();
        while ((camera_.is_running()) && (!done_)) {}
        camera_.stop();
        done_ = true;
    }
}

/// @brief Ignores events before start_ts and stop when we reach end_ts
/// @param ev_begin pointer on the first event
/// @param ev_end pointer on the last event
void EventProviderRaw::filter_ts(const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
    if (ev_begin != ev_end) {
        if (ev_begin->t < start_ts_) {
            // ignore until we reach at least start_ts
            return;
        }
        cb_(ev_begin, ev_end);
        if ((ev_end - 1)->t >= end_ts_) {
            // we went past end_ts_
            done_ = true;
        }
    }
}

/// @brief Starts streaming events
void EventProviderDat::start() {
    done_ = false;
    if (cb_ == nullptr) {
        done_ = true;
    } else {
        std::vector<EventCD> events;
        while ((!prod_->is_done()) && (current_ts_ < end_ts_) && (!done_)) {
            events.clear();
            current_ts_ += chunk_duration_;
            prod_->process_events(std::back_inserter(events), current_ts_);
            if (current_ts_ <= start_ts_) {
                continue;
            }

            if (!events.empty()) {
                cb_(&events[0], &events[0] + events.size());
            }
        }
        done_ = true;
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_ML_EVENT_PROVIDER_H
