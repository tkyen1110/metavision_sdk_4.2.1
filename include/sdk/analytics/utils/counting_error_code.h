/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_COUNTING_ERROR_CODE_H
#define METAVISION_SDK_ANALYTICS_COUNTING_ERROR_CODE_H

namespace Metavision {

/// @brief Alias type used for ErrorCode enums
/// @sa @ref CountingErrorCode
using ErrorCodeType = int;

/// @brief Enum that holds runtime error codes for metavision counting.
/// @sa @ref Metavision::CountingException
enum class CountingErrorCode : ErrorCodeType {
    /// Base metavision counting error
    Error = 0x300000,

    /// Errors related to invalid arguments
    InvalidArgument = Error | 0x01000,

    // CALIBRATION
    /// Invalid number of views
    InvalidNumViews = InvalidArgument | 0x1,
    /// Invalid counting input min size
    InvalidSize = InvalidArgument | 0x2,
    /// Invalid counting input average speed
    InvalidSpeed = InvalidArgument | 0x3,
    /// Invalid counting input average distance object camera
    InvalidDistance = InvalidArgument | 0x4,

    // COUNTING
    /// Invalid counting argument inactivity time
    InvalidInactivityTime = InvalidArgument | 0x5,
    /// Invalid counting argument notification sampling
    InvalidNotificationSampling = InvalidArgument | 0x6,
    /// Invalid input option
    InvalidOption = InvalidArgument | 0x7,
    /// Invalid counting data type
    InvalidDataType = InvalidArgument | 0x8,
    /// Invalid line position
    InvalidLinePosition = InvalidArgument | 0x9,
    /// Line not found (e.g., when removing a specific line)
    LineNotFound = InvalidArgument | 0xa,

    // GENERAL
    /// File does not exist
    FileDoesNotExist = InvalidArgument | 0xb,
    /// File extension is not the one expected
    WrongExtension = InvalidArgument | 0xc,
    /// Could not open file
    CouldNotOpenFile = InvalidArgument | 0xd,

    /// Counting runtime errors
    RuntimeError = Error | 0x02000,

    /// Modification of the Engine during run time.
    InvalidEngineModification = RuntimeError | 0x1,
};
} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_COUNTING_ERROR_CODE_H
