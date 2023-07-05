/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include "simple_video_writer.h"

SimpleVideoWriter::SimpleVideoWriter(int width, int height, int real_frame_period, const std::string &video_file_name) :
    SimpleVideoWriter(width, height, real_frame_period, (1'000'000.0f / real_frame_period), video_file_name) {}

SimpleVideoWriter::SimpleVideoWriter(int width, int height, int real_frame_period, float replay_fps,
                                     const std::string &video_file_name) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("The frame size should be greater than 0.");
    }

    const int fps = static_cast<int>(1'000'000.0 / real_frame_period);

    std::string full_file_name_ = video_file_name + "_" + std::to_string(fps) + "fps.avi";

    // Create VideoWriter object
    video_out_ = cv::VideoWriter(full_file_name_, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), replay_fps,
                                 cv::Size(width, height));
}

void SimpleVideoWriter::write_frame(const Metavision::timestamp ts, const cv::Mat &frame) {
    if (ts < save_time_from_ || ts > save_time_to_)
        return;

    // Record the video
    video_out_.write(frame);
}

void SimpleVideoWriter::set_write_range(const Metavision::timestamp &time_from, const Metavision::timestamp &time_to) {
    if (time_from > 0)
        save_time_from_ = time_from;
    if (time_to > 0)
        save_time_to_ = time_to;
}
