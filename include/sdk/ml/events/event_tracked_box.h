/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_TRACKED_BOX_H
#define METAVISION_SDK_ML_TRACKED_BOX_H

#include <sstream>
#include <cassert>
#include <cmath>
#include <algorithm>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief This class is used to output 2D tracked boxes
struct EventTrackedBox {
    /// @brief Creates an empty EventTrackedBox object
    /// @param t Timestamp
    /// @param x X coordinate of the top-left corner of the box
    /// @param y Y coordinate of the top-left corner of the box
    /// @param w Box's Width
    /// @param h Box's height
    /// @param class_id Box's class label identifier
    /// @param track_id Track identifier
    /// @param class_confidence Confidence score of the detection
    inline EventTrackedBox(timestamp t = 0, float x = 0.f, float y = 0.f, float w = 0.f, float h = 0.f,
                           unsigned int class_id = 0, unsigned int track_id = 0, float class_confidence = 0) noexcept :
        t(t),
        x(x),
        y(y),
        w(w),
        h(h),
        class_id(class_id),
        track_id(track_id),
        class_confidence(class_confidence),
        tracking_confidence(class_confidence),
        last_detection_update_time(t),
        nb_detections(1) {}

    /// @brief Creates an EventTrackedBox object from string from a csv format
    /// @param str Input string containing the tracked box in csv format
    /// The string should contain the following list attributes, separated by commas:
    /// t, class_id, track_id, x, y, w, h, class_confidence, tracking_confidence, last_detection_update_time,
    /// nb_detections
    EventTrackedBox(const std::string &str) {
        std::istringstream iss(str);
        char c;
        iss >> t;
        iss >> c;
        assert(c == EventTrackedBox::character_separator);
        iss >> class_id;
        iss >> c;
        assert(c == EventTrackedBox::character_separator);
        iss >> track_id;
        iss >> c;
        assert(c == EventTrackedBox::character_separator);
        iss >> x;
        iss >> c;
        assert(c == EventTrackedBox::character_separator);
        iss >> y;
        iss >> c;
        assert(c == EventTrackedBox::character_separator);
        iss >> w;
        iss >> c;
        assert(c == EventTrackedBox::character_separator);
        iss >> h;
        iss >> c;
        assert(c == EventTrackedBox::character_separator);
        iss >> class_confidence;
        iss >> c;
        assert(c == EventTrackedBox::character_separator);
        iss >> tracking_confidence;
        iss >> c;
        assert(c == EventTrackedBox::character_separator);
        iss >> last_detection_update_time;
        iss >> c;
        assert(c == EventTrackedBox::character_separator);
        iss >> nb_detections;
    }

    /// @brief Writes the tracked box into a csv format
    /// @param output Stream to write the tracked box
    void write_csv_line(std::ostream &output) const {
        output << t << EventTrackedBox::character_separator << class_id << EventTrackedBox::character_separator
               << track_id << EventTrackedBox::character_separator << x << EventTrackedBox::character_separator << y
               << EventTrackedBox::character_separator << w << EventTrackedBox::character_separator << h
               << EventTrackedBox::character_separator << class_confidence << EventTrackedBox::character_separator
               << tracking_confidence << EventTrackedBox::character_separator << last_detection_update_time
               << EventTrackedBox::character_separator << nb_detections << std::endl;
    }

    /// @brief Serializes an EventBbox into a stream
    /// @param output Stream
    /// @param e EventBbox to be serialized
    /// @return Stream provided as input
    friend std::ostream &operator<<(std::ostream &output, const EventTrackedBox &e) {
        output << "EventTrackedBox: ("
               << "t: " << e.t << "   "
               << "x: " << e.x << "   "
               << "y: " << e.y << "   "
               << "w: " << e.w << "   "
               << "h: " << e.h << "   "
               << "class_id: " << e.class_id << "   "
               << "track_id: " << e.track_id << "   "
               << "class_confidence: " << e.class_confidence << "   "
               << "tracking_confidence: " << e.tracking_confidence << "   "
               << "last_detection_update_time: " << e.last_detection_update_time << "   "
               << "nb_detections: " << e.nb_detections << ")";
        return output;
    }

    /// @brief Returns csv format description
    /// @return String describing csv format
    static std::string getHeader() {
        std::ostringstream oss;
        oss << "#t" << character_separator << "class_id" << character_separator << "track_id" << character_separator
            << "x" << character_separator << "y" << character_separator << "w" << character_separator << "h"
            << character_separator << "class_confidence" << character_separator << "tracking_confidence"
            << character_separator << "last_detection_update_time" << character_separator << "nb_detections"
            << std::endl;
        return oss.str();
    }

    /// @brief Updates the last detection timestamp and compute a new track confidence value
    /// @param t Timestamp of the new detection
    /// @param detection_confidence Detection confidence value
    /// @param similarity_box_track Weight applied on the detection to compute track detection
    inline void set_last_detection_update(timestamp t, float detection_confidence = 0.5f,
                                          float similarity_box_track = 1.0f) {
        last_detection_update_time = t;
        class_confidence           = detection_confidence;
        tracking_confidence += detection_confidence * similarity_box_track;
        tracking_confidence = std::min(tracking_confidence, 1.f);
    }

    // attributes
    timestamp t;                              ///< Timestamp of the box
    float x;                                  ///< X position of the bounding box
    float y;                                  ///< Y position of the bounding box
    float w;                                  ///< Width of the bounding box
    float h;                                  ///< Height of the bounding box
    unsigned int class_id;                    ///< bounding box's class id
    int track_id;                             ///< Track identifier
    float class_confidence;                   ///< Confidence of the detection
    float tracking_confidence;                ///< Confidence computed from previous detection and matching
    timestamp last_detection_update_time = 0; ///< Time of last update of the detection.
    int nb_detections;                        ///< Number of time this box have been seen

    const static char character_separator = ','; ///< CSV format separator
};

} // namespace Metavision

#endif // METAVISION_SDK_ML_TRACKED_BOX_H
