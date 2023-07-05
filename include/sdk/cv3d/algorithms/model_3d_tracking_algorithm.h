/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV3D_MODEL_3D_TRACKING_ALGORITHM_H
#define METAVISION_SDK_CV3D_MODEL_3D_TRACKING_ALGORITHM_H

#include <string>
#include <set>

#include "metavision/sdk/cv/utils/camera_geometry_base.h"
#include "metavision/sdk/cv/utils/gauss_newton_solver.h"
#include "metavision/sdk/cv3d/utils/model_3d.h"
#include "metavision/sdk/cv3d/utils/edge_ls_problem.h"

namespace Metavision {

template<typename T>
class MostRecentTimestampBufferT;

using MostRecentTimestampBuffer = MostRecentTimestampBufferT<timestamp>;
using CameraGeometry32f         = CameraGeometryBase<float>;

/// @brief Algorithm that estimates the 6 DOF pose of a 3D model by tracking its edges in an events stream
///
/// Support points are sampled along the 3D model's visible edges and tracked in a time surface in which the events
/// stream has been accumulated. Matches are looked for in the time surface within a fixed radius and a given
/// accumulation time. A weight is then attributed to every support point according to the timestamp of the event
/// they matched with. Finally, the pose is estimated using a weighted least squares to minimize the orthogonal distance
/// between the matches and their corresponding reprojected edge.
///
/// The accumulation time used for matching can vary depending on how the algorithm is called. Indeed, the algorithm
/// computes the accumulation time by maintaining a sliding buffer of the last N pose computation timestamps.
/// As a result, the accumulation time will be fixed when the algorithm is called every N us and varying when called
/// every N events. The latter is more interesting because, in that case, the accumulation will adapt to the motion of
/// the camera. In case of fast motion, the tracking will restrict the matching to very recent events making it more
/// robust to noise. Whereas in case of slow motion, the tracking will allow matching older events.
class Model3dTrackingAlgorithm {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// @brief Parameters used by the model 3d tracker algorithm
    struct Parameters {
        Parameters(){}; // We have to define a default constructor like this because of a bug in GCC & clang

        std::uint32_t search_radius_      = 3;  ///< Radius in which matches are looked for for each support point
        std::uint32_t support_point_step_ = 10; ///< Distance, in pixels in the distorted image, between two support
                                                ///< points
        std::uint32_t n_last_poses_    = 5;     ///< Number of past poses to consider to compute the accumulation time
        timestamp default_acc_time_us_ = 3000;  ///< Default accumulation time used when the tracking is starting (i.e.
                                                /// the N last poses have not been estimated yet)
        float oldest_weight_      = 0.1f;       ///< Weight attributed to the oldest matches
        float most_recent_weight_ = 1.f;        ///< Weight attributed to the more recent matches
    };

    /// @brief Constructor
    /// @param cam_geometry Camera geometry instance allowing mapping coordinates from camera to image (and vice versa)
    /// @param model 3D model to track
    /// @param time_surface Time surface instance in which the events stream is accumulated
    /// @param params Algorithm's parameters
    Model3dTrackingAlgorithm(const CameraGeometry32f &cam_geometry, const Model3d &model,
                             MostRecentTimestampBuffer &time_surface, const Parameters &params = Parameters());

    /// @brief Initializes the tracking by setting a camera's prior pose
    /// @param ts Timestamp at which the pose was estimated
    /// @param T_c_w Camera's prior pose
    void set_previous_camera_pose(timestamp ts, const Eigen::Matrix4f &T_c_w);

    /// @brief Tries to track the 3D model from the input events buffer
    /// @tparam InputIt Read-Only input event iterator type.
    /// @param[in] it_begin Iterator to the first input event
    /// @param[in] it_end Iterator to the past-the-end event
    /// @param[out] T_c_w Camera's pose if the tracking has succeeded
    /// @return True if the tracking has succeeded, false otherwise
    template<typename InputIt>
    bool process_events(InputIt it_begin, InputIt it_end, Eigen::Matrix4f &T_c_w);

private:
    bool process_internal();

    using GNReport = GaussNewton::Report<float>;

    const CameraGeometry32f &cam_geometry_;
    const Model3d &model_3d_;
    MostRecentTimestampBuffer &time_surface_;
    Parameters params_;
    Eigen::Matrix4f T_c_w_;
    EdgeLSProblem edge_ls_problem_;
    GNReport gn_report_;
    std::set<size_t> visible_edges_;
    EdgeDataAssociationVector data_associations_;
    std::vector<cv::Matx21f> match_candidates_;
    EdgeDataAssociationVector filtered_data_associations_;
    std::vector<timestamp> pose_ts_;
    size_t ts_idx_;
};
} // namespace Metavision

#include "metavision/sdk/cv3d/algorithms/detail/model_3d_tracking_algorithm_impl.h"

#endif // METAVISION_SDK_CV3D_MODEL_3D_TRACKING_ALGORITHM_H
