/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_SIMPLE_VIDEO_WRITER_H
#define METAVISION_SDK_ANALYTICS_SIMPLE_VIDEO_WRITER_H

#include <fstream>
#include <string>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <metavision/sdk/base/utils/timestamp.h>

/// @brief Class representing a simple video writer
class SimpleVideoWriter {
public:
    /// @brief Constructor. With this version, the generated video will be replayed at real time.
    ///        To generate a "slow motion" video, use the version that specifies replay_fps.
    ///
    /// @param width Width of the image
    /// @param height Height of the image
    /// @param real_frame_period Real duration in us elapsed between two frames
    /// @param video_file_name Output video filename
    SimpleVideoWriter(int width, int height, int real_frame_period, const std::string &video_file_name);

    /// @brief Constructor. With this version, the generated video will be replayed @p replay_fps frames
    ///        per second.
    ///
    /// @param width Width of the image
    /// @param height Height of the image
    /// @param real_frame_period Real duration in us elapsed between two frames
    /// @param replay_fps Replay speed
    /// @param video_file_name Output video filename
    SimpleVideoWriter(int width, int height, int real_frame_period, float replay_fps,
                      const std::string &video_file_name);

    /// @brief Writes a frame in the output video
    ///
    /// @param ts Current timestamp
    /// @param frame Frame to be written
    void write_frame(const Metavision::timestamp ts, const cv::Mat &frame);

    /// @brief Specifies the time slice that should be recorded
    /// @param time_from Beginning of the time slice
    /// @param time_to End of the time slice
    void set_write_range(const Metavision::timestamp &time_from, const Metavision::timestamp &time_to);

private:
    // Record from the time
    Metavision::timestamp save_time_from_ = 0;
    // Record up to the time
    Metavision::timestamp save_time_to_ = std::numeric_limits<Metavision::timestamp>::max();

    // Video recorder
    cv::VideoWriter video_out_;
};

#endif // METAVISION_SDK_ANALYTICS_SIMPLE_VIDEO_WRITER_H
