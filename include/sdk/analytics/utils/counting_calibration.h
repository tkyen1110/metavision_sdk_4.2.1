/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_COUNTING_CALIBRATION_H
#define METAVISION_SDK_ANALYTICS_COUNTING_CALIBRATION_H

namespace Metavision {

/// @brief Class representing the counting calibration
class CountingCalibration {
public:
    /// @brief Struct storing the calibration results
    struct Results {
        int cluster_ths;
        int accumulation_time_us;
    };

    /// @brief Finds optimal parameters for the counting algorithm
    /// @param width Sensor's width in pixels
    /// @param height Sensor's height in pixels
    /// @param object_min_size Approximate largest dimension of the smallest object (in mm).
    /// The value must be positive. It will be refined during the calibration
    /// @param object_average_speed Approximate average speed of an object to count (in m/s). It will be
    /// refined during the calibration.
    /// @param distance_object_camera Average distance between the flow of objects to count and the camera (in mm)
    /// Camera must look perpendicular to the object falling plane. It will be refined during the calibration
    /// @param horizontal_fov Horizontal field of view (half of the solid angle perceived by the sensor along the
    /// horizontal axis, in degrees)
    /// @param vertical_fov Vertical field of view (half of the solid angle perceived by the sensor along the vertical
    /// axis, in degrees)
    /// @param travelled_pix_distance_during_acc_time Distance (in pixels) travelled during the accumulation time
    static Results calibrate(int width, int height, float object_min_size = 5, float object_average_speed = 5,
                             float distance_object_camera = 300, float horizontal_fov = 56.f, float vertical_fov = 44.f,
                             int travelled_pix_distance_during_acc_time = 9);
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_COUNTING_CALIBRATION_H
