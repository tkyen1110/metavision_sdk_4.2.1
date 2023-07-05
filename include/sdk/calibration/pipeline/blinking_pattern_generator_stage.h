/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CALIBRATION_BLINKING_FRAME_GENERATOR_H
#define METAVISION_SDK_CALIBRATION_BLINKING_FRAME_GENERATOR_H

#include <boost/filesystem.hpp>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/base/utils/object_pool.h>
#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/core/pipeline/pipeline.h>
#include <metavision/sdk/calibration/algorithms/blinking_frame_generator_algorithm.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace Metavision {

/// @brief Stage that accumulates events and keeps pixels which were activated with both polarities during the
/// accumulating period, if enough of them are found.
///
/// Produces a binary (or grayscale if enable_event_count is set to True)
/// frame representing blinking pixels (0 or 255). If enable_event_count is True, the value is the
/// number of events during the accumulation period.
///   - Input : buffer of events                        : EventBufferPtr
///   - Output: timestamped frame (Blinking Chessboard) : FrameData
///
/// @ref BaseStage
class BlinkingFrameGeneratorStage : public Metavision::BaseStage {
public:
    using EventBuffer    = std::vector<Metavision::EventCD>;
    using EventBufferPtr = Metavision::SharedObjectPool<EventBuffer>::ptr_type;

    using FramePool = Metavision::SharedObjectPool<cv::Mat>;
    using FramePtr  = FramePool::ptr_type;
    using FrameData = std::pair<Metavision::timestamp, FramePtr>;

    /// @brief Constructor
    /// @param width Image width
    /// @param height Image height
    /// @param blinking_config Blinking frame generator algorithm configuration
    /// @param output_images_dir_path Output image directory. If equal to "", images will not be saved.
    BlinkingFrameGeneratorStage(int width, int height,
                                const Metavision::BlinkingFrameGeneratorAlgorithmConfig &blinking_config,
                                const std::string &output_images_dir_path = "") {
        blink_detector_ = std::make_unique<Metavision::BlinkingFrameGeneratorAlgorithm>(width, height, blinking_config);

        export_frames_ = (output_images_dir_path != "");
        if (export_frames_) {
            base_frames_path_ = (boost::filesystem::path(output_images_dir_path) / "pattern_").string();
        }

        set_consuming_callback([this](const boost::any &data) {
            try {
                auto buffer = boost::any_cast<EventBufferPtr>(data);
                if (buffer->empty())
                    return;
                successful_cb_ = false;
                blink_detector_->process_events(buffer->cbegin(), buffer->cend());
                if (!successful_cb_)
                    produce(std::make_pair(buffer->crbegin()->t, FramePtr())); // Temporal marker

            } catch (boost::bad_any_cast &c) { MV_LOG_ERROR() << c.what(); }
        });

        frame_pool_ = FramePool::make_bounded(2);
        frame_id_   = 1;
        blink_detector_->set_output_callback([this](Metavision::timestamp ts, cv::Mat &blinking_img) {
            successful_cb_        = true;
            auto output_frame_ptr = frame_pool_.acquire();

            produce(std::make_pair(ts, output_frame_ptr));
            if (export_frames_) {
                std::stringstream ss;
                ss << base_frames_path_ << frame_id_++ << ".png";
                cv::imwrite(ss.str(), blinking_img);
            }
            cv::swap(blinking_img, *output_frame_ptr);
        });
    }

private:
    FramePool frame_pool_;
    std::unique_ptr<Metavision::BlinkingFrameGeneratorAlgorithm> blink_detector_;
    bool successful_cb_;
    int frame_id_;

    bool export_frames_;
    std::string base_frames_path_;
};
} // namespace Metavision
#endif