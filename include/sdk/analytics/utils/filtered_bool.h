/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_FILTERED_BOOL_H
#define METAVISION_SDK_ANALYTICS_FILTERED_BOOL_H

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Class implementing a boolean variable that's updated in time, and whose value only changes when the new value
/// has been validated for a certain, configurable, time
class FilteredBool {
public:
    /// @brief Default constructor
    FilteredBool() = default;

    /// @brief Constructor
    /// @param initial_value Initial value
    /// @param rise_delay_us Delay to validate a rising edge. The internal value will change from "false" to "true" only
    /// if the variable is updated to "true" for a duration >= rise_delay_us
    /// @param fall_delay_us Delay to validate a falling edge. The internal value will change from "true" to "false"
    /// only if the variable is updated to "false" for a duration  >= fall_delay_us
    FilteredBool(bool initial_value, timestamp rise_delay_us, timestamp fall_delay_us);

    /// @brief Updates the observed value
    /// @param ts Timestamp of the update. Successive calls must have increasing values of @p ts
    /// @param value Observed value to update
    /// @throw std::runtime_error if successive values of @p ts are not increasing
    void update(timestamp ts, bool value);

    /// @brief Returns the filtered value
    /// @return The filtered value
    bool value() const;

    /// @brief Returns the unfiltered value
    /// @return The last value updated, regardless of if it has been confirmed
    bool raw_value() const;

    /// @brief Gets the timestamp of the last time that a call to @ref update() produced a rising edge that was then
    /// validated by >= rise_delay_us
    /// @return The last rising edge timestamp
    timestamp last_rising_edge_ts() const;

    /// @brief Gets the timestamp of the last time that a call to @ref update() produced a falling edge that was then
    /// validated by >= fall_delay_us
    /// @return The last falling edge timestamp
    timestamp last_falling_edge_ts() const;

private:
    bool value_{false};                ///< Internal filtered value of the signal
    bool previous_input_value_{false}; ///< Internal instantaneous value of the signal
    timestamp previous_update_ts_{-1}; ///< Last time the value was updated
    timestamp rise_delay_{0};          ///< Delay to validate a rising edge
    timestamp fall_delay_{0};          ///< Delay to validate a falling edge

    timestamp last_rising_edge_ts_{-1};  ///< Last time the value changed from false to true
    timestamp last_falling_edge_ts_{-1}; ///< Last time the value changed from true to false

    /// @brief Last time the value changed from false to true and remained true for >= rise_delay_us
    timestamp last_valid_rising_edge_ts_{-1};

    /// @brief Last time the value changed from true to false and remained false for >= fall_delay_us
    timestamp last_valid_falling_edge_ts_{-1};
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_FILTERED_BOOL_H