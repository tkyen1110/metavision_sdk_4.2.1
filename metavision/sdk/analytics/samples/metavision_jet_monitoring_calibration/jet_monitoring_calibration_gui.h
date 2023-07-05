/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef JET_MONITORING_CALIBRATION_GUI_H
#define JET_MONITORING_CALIBRATION_GUI_H

#include <mutex>
#include <opencv2/opencv.hpp>

#include <metavision/sdk/ui/utils/mt_window.h>

namespace Metavision {

/// @brief Class acting as a Graphical User Interface for the Jet Monitoring calibration tool
class JetMonitoringCalibrationGUI {
public:
    /// @brief Constructor
    /// @param width Sensor's width (in pixels)
    /// @param height Sensor's height (in pixels)
    /// @param transpose_output_rois Set to true if the nozzle is firing jets vertically in the FOV
    JetMonitoringCalibrationGUI(int width, int height, bool transpose_output_rois);

    /// @brief Indicates whether the window has been asked to close
    /// @return True if the window should close, False otherwise
    bool should_close() const;

    /// @brief Updates the background CD frame by swapping it if we are in "Play" mode, does nothing otherwise ("Pause"
    /// mode)
    /// @param cd_frame CD frame to be swapped
    void swap_cd_frame_if_required(cv::Mat &cd_frame);

    /// @brief Updates the display
    void update();

private:
    void draw_baseline(int y, bool final_state = true);

    void draw_jet_rois(int x, int y, const cv::Point &corner_offset, bool final_state = true);

    void draw_camera_roi(int x, int y, const cv::Point &corner_offset, bool final_state = true);

    void print_help_msg(const std::vector<std::string> &help_msg);

    cv::Rect get_roi(int x_ref, int y_ref, const cv::Point &corner_offset, bool transpose = false) const;

    enum State {
        NONE,
        BASELINE,
        CAMERA_ROI,
        JET_ROI,
    };

    bool transpose_output_rois_;
    int width_, height_;

    // Events frames
    bool update_cd_img_;
    cv::Mat front_img_;
    cv::Mat last_cd_img_;
    std::mutex process_mutex_;

    // Horizontal baseline
    int baseline_y_;

    // Jet ROI
    int jet_x_;
    cv::Point jet_corner_offset_;

    // Camera ROI
    int cam_x_;
    cv::Point cam_corner_offset_;

    //  State
    bool is_initializing_roi_;
    State state_;

    // Window
    MTWindow window_;
    cv::Point last_mouse_pos_;

    // Help message
    static constexpr int FONT_FACE     = cv::FONT_HERSHEY_SIMPLEX; ///< Font used for text rendering
    static constexpr double FONT_SCALE = 0.5;                      ///< Font scale used for text rendering
    static constexpr int THICKNESS     = 1;                        ///< Line thickness used for text rendering
    static constexpr int MARGIN        = 3;                        ///< Additional space used for text rendering
    cv::Point help_msg_text_pos_;                                  ///< Position of the help message in the image
    int help_text_height_;                                         ///< Maximum text height

    const cv::Vec3b color_txt_          = cv::Vec3b(219, 226, 228);
    const cv::Vec3b color_tmp_          = cv::Vec3b(0, 255, 255);
    const cv::Vec3b color_baseline_     = cv::Vec3b(221, 207, 193);
    const cv::Vec3b color_roi_          = cv::Vec3b(118, 114, 255);
    const cv::Vec3b color_bg_noise_roi_ = cv::Vec3b(201, 126, 64);
};

} // namespace Metavision

#endif // JET_MONITORING_CALIBRATION_GUI_H
