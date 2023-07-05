/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <metavision/sdk/base/utils/sdk_log.h>

#include "jet_monitoring_calibration_gui.h"

namespace Metavision {

JetMonitoringCalibrationGUI::JetMonitoringCalibrationGUI(int width, int height, bool transpose_output_rois) :
    transpose_output_rois_(transpose_output_rois),
    width_(width),
    height_(height),
    update_cd_img_(true),
    baseline_y_(height / 2), // Start with a baseline in the middle of the image
    jet_x_(-1),
    jet_corner_offset_(cv::Point(-1, -1)),
    cam_x_(-1),
    cam_corner_offset_(cv::Point(-1, -1)),
    is_initializing_roi_(false),
    state_(State::NONE),
    window_("Jet Monitoring Calibration", width, height, BaseWindow::RenderMode::BGR),
    last_mouse_pos_(cv::Point(-1, -1)) {
    front_img_.create(height, width, CV_8UC3);
    front_img_.setTo(cv::Scalar::all(0));

    // Help message
    int base_line        = 0;
    const auto text_size = cv::getTextSize("Jet Monitoring", FONT_FACE, FONT_SCALE, THICKNESS, &base_line);
    help_msg_text_pos_   = cv::Point(MARGIN, text_size.height + MARGIN);
    help_text_height_    = text_size.height + base_line;

    // Cursor callback
    window_.set_cursor_pos_callback([this](double x, double y) {
        int window_width, window_height;
        window_.get_size(window_width, window_height);

        // The window may have been resized meanwhile. So we map the coordinates to the original window's size.
        const auto mapped_x = static_cast<int>(x * width_ / window_width);
        const auto mapped_y = static_cast<int>(y * height_ / window_height);
        last_mouse_pos_     = cv::Point(mapped_x, mapped_y);
    });

    // Mouse callback
    window_.set_mouse_callback([this](Metavision::UIMouseButton button, Metavision::UIAction action, int mods) {
        if (button != Metavision::UIMouseButton::MOUSE_BUTTON_LEFT)
            return; // Only left click is being used

        if (action == Metavision::UIAction::PRESS) {
            is_initializing_roi_ = false;
            if (state_ == State::JET_ROI)
                jet_x_ = last_mouse_pos_.x;
            else if (state_ == State::CAMERA_ROI)
                cam_x_ = last_mouse_pos_.x;
        } else if (action == Metavision::UIAction::RELEASE) {
            switch (state_) {
            case State::BASELINE:
                baseline_y_ = last_mouse_pos_.y;
                break;
            case State::JET_ROI:
                jet_corner_offset_.x = last_mouse_pos_.x - jet_x_;
                jet_corner_offset_.y = last_mouse_pos_.y - baseline_y_;
                break;
            case State::CAMERA_ROI:
                cam_corner_offset_.x = last_mouse_pos_.x - cam_x_;
                cam_corner_offset_.y = last_mouse_pos_.y - baseline_y_;
                break;
            }
            state_ = State::NONE; // Baseline or ROI has been validated, go back to "None" state
        }
    });

    // Keyboard callback
    window_.set_keyboard_callback([this](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action,
                                         int mods) {
        if (action != Metavision::UIAction::RELEASE || state_ != State::NONE)
            return;

        switch (key) {
        case UIKeyEvent::KEY_SPACE:
            update_cd_img_ = !update_cd_img_; // Play/pause events
            break;
        case UIKeyEvent::KEY_B:
            state_ = State::BASELINE;
            break;
        case UIKeyEvent::KEY_C:
            state_               = State::CAMERA_ROI;
            is_initializing_roi_ = true;
            break;
        case UIKeyEvent::KEY_J:
            state_               = State::JET_ROI;
            is_initializing_roi_ = true;
            break;
        case UIKeyEvent::KEY_ENTER:
            if (jet_x_ < 0 && cam_x_ < 0)
                MV_LOG_INFO() << "No ROIs have been defined. Press C or J to define them.";
            else {
                MV_LOG_INFO() << "----------------";
                if (transpose_output_rois_)
                    MV_LOG_INFO() << "(ROIs below are expressed in the original, non-transposed, image frame)";
                if (jet_x_ > 0) {
                    const cv::Rect jet_roi = get_roi(jet_x_, baseline_y_, jet_corner_offset_, transpose_output_rois_);
                    MV_LOG_INFO() << " --detection-roi" << jet_roi.x << jet_roi.y << jet_roi.width << jet_roi.height;
                }
                if (cam_x_ > 0) {
                    const cv::Rect cam_roi = get_roi(cam_x_, baseline_y_, cam_corner_offset_, transpose_output_rois_);
                    MV_LOG_INFO() << " --camera-roi" << cam_roi.x << cam_roi.y << cam_roi.width << cam_roi.height;
                }
                MV_LOG_INFO() << "----------------";
            }
            break;
        case UIKeyEvent::KEY_ESCAPE:
        case UIKeyEvent::KEY_Q:
            window_.set_close_flag();
            break;
        }
    });
};

bool JetMonitoringCalibrationGUI::should_close() const {
    return window_.should_close();
}

void JetMonitoringCalibrationGUI::swap_cd_frame_if_required(cv::Mat &cd_frame) {
    std::lock_guard<std::mutex> lock(process_mutex_);
    if (update_cd_img_)
        cv::swap(cd_frame, last_cd_img_);
}

void JetMonitoringCalibrationGUI::update() {
    {
        std::lock_guard<std::mutex> lock(process_mutex_);
        last_cd_img_.copyTo(front_img_);
    }

    switch (state_) {
    case State::NONE: {
        static const std::vector<std::string> help_msg = {
            "Press 'Space' to play/pause events", "Press 'B' to define the baseline",
            "Press 'C' to define the Camera ROI", "Press 'J' to define the Jet ROI",
            "Press 'Enter' to print ROIs",        "Press 'Q' or 'Escape' to exit"};
        print_help_msg(help_msg);
        draw_baseline(baseline_y_);
        draw_jet_rois(jet_x_, baseline_y_, jet_corner_offset_);
        draw_camera_roi(cam_x_, baseline_y_, cam_corner_offset_);
    } break;
    case State::BASELINE: {
        static const std::vector<std::string> help_msg = {"Left click when aligned with the jet direction"};
        print_help_msg(help_msg);
        draw_baseline(last_mouse_pos_.y, false);
        draw_jet_rois(jet_x_, last_mouse_pos_.y, jet_corner_offset_, false);
        draw_camera_roi(cam_x_, last_mouse_pos_.y, cam_corner_offset_, false);
    } break;
    case State::CAMERA_ROI: {
        static const std::vector<std::string> help_msg = {"Click and drag to define the Camera ROI"};
        print_help_msg(help_msg);
        draw_baseline(baseline_y_);
        draw_jet_rois(jet_x_, baseline_y_, jet_corner_offset_);
        if (is_initializing_roi_)
            cv::line(front_img_, cv::Point(last_mouse_pos_.x, 0), cv::Point(last_mouse_pos_.x, height_ - 1),
                     color_tmp_);
        else
            draw_camera_roi(cam_x_, baseline_y_, cv::Point(last_mouse_pos_.x - cam_x_, last_mouse_pos_.y - baseline_y_),
                            false);
    } break;
    case State::JET_ROI: {
        static const std::vector<std::string> help_msg = {"Click and drag to define the Jet ROI and",
                                                          "its two surrounding Background Activity ROIs"};
        print_help_msg(help_msg);
        draw_baseline(baseline_y_);
        draw_camera_roi(cam_x_, baseline_y_, cam_corner_offset_);
        if (is_initializing_roi_)
            cv::line(front_img_, cv::Point(last_mouse_pos_.x, 0), cv::Point(last_mouse_pos_.x, height_ - 1),
                     color_tmp_);
        else
            draw_jet_rois(jet_x_, baseline_y_, cv::Point(last_mouse_pos_.x - jet_x_, last_mouse_pos_.y - baseline_y_),
                          false);
    } break;
    }

    // Display
    window_.show_async(front_img_);
}

void JetMonitoringCalibrationGUI::draw_baseline(int y, bool final_state) {
    cv::line(front_img_, cv::Point(0, y), cv::Point(width_ - 1, y), (final_state ? color_baseline_ : color_tmp_));
}

void JetMonitoringCalibrationGUI::draw_jet_rois(int x, int y, const cv::Point &corner_offset, bool final_state) {
    // Draw ROI only if it's defined
    if (jet_x_ < 0)
        return;

    cv::Rect jet_roi = get_roi(x, y, corner_offset);
    cv::rectangle(front_img_, jet_roi, (final_state ? color_roi_ : color_tmp_), 1, 4);

    jet_roi.y += jet_roi.height;
    cv::rectangle(front_img_, jet_roi, (final_state ? color_bg_noise_roi_ : color_tmp_), 1, 4);
    jet_roi.y -= 2 * jet_roi.height;
    cv::rectangle(front_img_, jet_roi, (final_state ? color_bg_noise_roi_ : color_tmp_), 1, 4);
}

void JetMonitoringCalibrationGUI::draw_camera_roi(int x, int y, const cv::Point &corner_offset, bool final_state) {
    // Draw ROI only if it's defined
    if (cam_x_ < 0)
        return;

    const cv::Rect cam_roi = get_roi(x, y, corner_offset);
    cv::rectangle(front_img_, cam_roi, (final_state ? color_roi_ : color_tmp_), 1, 4);
}

void JetMonitoringCalibrationGUI::print_help_msg(const std::vector<std::string> &help_msg) {
    cv::Point text_pos = help_msg_text_pos_;
    for (const auto &s : help_msg) {
        cv::putText(front_img_, s, text_pos, FONT_FACE, FONT_SCALE, color_txt_, THICKNESS, cv::LINE_AA);
        text_pos.y += help_text_height_ + MARGIN;
    }
}

cv::Rect JetMonitoringCalibrationGUI::get_roi(int x_ref, int y_ref, const cv::Point &corner_offset,
                                              bool transpose) const {
    // ROI is centered around the baseline
    cv::Rect roi(x_ref, y_ref - std::abs(corner_offset.y), std::abs(corner_offset.x), 2 * std::abs(corner_offset.y));
    if (corner_offset.x < 0)
        roi.x += corner_offset.x;

    // Transpose if needed
    if (transpose) {
        std::swap(roi.x, roi.y);
        std::swap(roi.width, roi.height);
    }
    return roi;
}

} // namespace Metavision
