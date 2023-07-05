/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_COUNTING_EXCEPTION_H
#define METAVISION_SDK_ANALYTICS_COUNTING_EXCEPTION_H

#include <system_error>
#include <memory>

#include "metavision/sdk/analytics/utils/counting_error_code.h"

namespace Metavision {

/// @brief Class for all exceptions thrown from Metavision counting.
/// @sa http://www.cplusplus.com/reference/system_error/system_error/
/// @sa http://en.cppreference.com/w/cpp/error/error_code
class CountingException : public std::system_error {
public:
    /// @brief Creates an exception of type e with default error message
    /// @param e Counting error code
    /// @sa @ref CountingErrorCode
    CountingException(CountingErrorCode e);

    /// @brief Creates an exception of type e with precise error description contained in additional_info
    /// @param e Counting error code
    /// @param additional_info Message containing information about the error
    /// @sa @ref CountingErrorCode
    CountingException(CountingErrorCode e, const std::string &additional_info);
};
} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_COUNTING_EXCEPTION_H
