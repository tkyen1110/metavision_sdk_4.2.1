/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_TRACKING_ALGORITHM_CONFIG_H
#define METAVISION_SDK_ANALYTICS_TRACKING_ALGORITHM_CONFIG_H

#include <istream>
#include <limits>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Structure used to configure the @ref TrackingAlgorithm.
///
/// There are 4 main parts:
/// 1. Clustering using one of the methods indexed by @ref TrackingConfig::ClusterMaker
/// 2. Data association using one of the methods indexed by @ref TrackingConfig::DataAssociation
/// 3. Tracker initialization using the method indexed by @ref TrackingConfig::Initializer
/// 4. Tracking using one of the methods indexed by @ref TrackingConfig::Tracker
struct TrackingConfig {
    bool print_timings_ = false; ///< If enabled, displays a profiling summary.

    /// @brief Defines the type of cluster maker used by the @ref TrackingAlgorithm
    /// to build clusters from input events.
    enum class ClusterMaker {
        SIMPLE_GRID, ///< Cluster making is based on a regular grid:
                     ///< - The camera FOV is divided into elementary cells using a regular grid;
                     ///<   the size of the cells is @ref cell_width_ x @ref cell_height_.
                     ///< - For each cell, if the number of events in the cell for the given time-slice
                     ///<   @ref cell_deltat_ is larger than the @ref activation_threshold_,
                     ///<   then the cell is considered as active.
                     ///< - Active cells of a given time-slice are connected into clusters.
        MEDOID_SHIFT ///< Cluster making is based on spatial and temporal distances between neighboring events.
                     ///< If the spatial and temporal distances between the event and its neighboring event
                     ///< are smaller than the thresholds @ref medoid_shift_spatial_dist_ and
                     ///< @ref medoid_shift_temporal_dist_, then the event goes to the same cluster as its neighbour,
                     ///< otherwise, it creates a new cluster.
    };

    ClusterMaker cluster_maker_ = ClusterMaker::SIMPLE_GRID; ///< Type of cluster maker to use

    // Grid cluster related parameters.
    int cell_width_           = 10;   ///< Cell's width (in pixels) in @ref ClusterMaker::SIMPLE_GRID.
    int cell_height_          = 10;   ///< Cell's height (in pixels) in @ref ClusterMaker::SIMPLE_GRID.
    timestamp cell_deltat_    = 1000; ///< Time-slice (in us) used to decide if a cell is active in
                                      ///< @ref ClusterMaker::SIMPLE_GRID.
    int activation_threshold_ = 5;    ///< Minimum number of events needed to activate the cell in
                                      ///< @ref ClusterMaker::SIMPLE_GRID.

    // Medoid shift cluster related parameters.
    /// Maximum spatial distance (using Manhattan distance) for two events to be in the same cluster in
    /// @ref ClusterMaker::MEDOID_SHIFT.
    float medoid_shift_spatial_dist_ = 5;
    /// Maximum temporal distance for two events to be in the same cluster in @ref ClusterMaker::MEDOID_SHIFT.
    timestamp medoid_shift_temporal_dist_ = 10000;
    /// Minimum width and height for a cluster to be considered valid in @ref ClusterMaker::MEDOID_SHIFT
    /// and given to the tracking engine.
    int medoid_shift_min_size_ = 2;

    /// @brief Defines the type of data association used by the @ref TrackingAlgorithm
    /// to associate each cluster to one tracker.
    enum class DataAssociation {
        NEAREST, ///< Data association is based on the nearest tracker.
        IOU      ///< Data association is based the largest IOU (Intersection Over Union) area between the tracker and
                 ///< the cluster. If no IOU association is available but the distance between the tracker and
                 ///< the cluster is smaller than @ref iou_max_dist_, then the cluster is associated using the nearest
                 ///< criterion. The IOU criterion has the priority over the distance criterion.
    };

    DataAssociation data_association_ = DataAssociation::IOU; ///< Type of data association to use.

    // Nearest data association related parameters.
    /// Maximum distance between the cluster's centroid and the tracker's position (in pixels) used in
    /// @ref DataAssociation::NEAREST.
    double max_dist_ = 150.;

    // IOU data association related parameters.
    /// Maximum distance between the cluster's centroid and the tracker's position (in pixels) used in
    /// @ref DataAssociation::IOU. The distance criterion is used only if IOU association is not available.
    double iou_max_dist_ = 150.;

    /// @brief Defines the type of initializer used by the @ref TrackingAlgorithm
    /// to initialize new trackers and make bounding box proposals from input cluster/events.
    /// Only one method is implemented for now.
    enum class Initializer { SIMPLE };

    // Simple initializer related parameters.
    Initializer initializer_         = Initializer::SIMPLE; ///< Type of tracker initializer to use.
    int simple_initializer_min_size_ = 0;
    int simple_initializer_max_size_ = std::numeric_limits<std::uint16_t>::max();

    /// @brief Defines the type of motion model used by the @ref TrackingAlgorithm
    /// to predict the positions of trackers.
    enum class MotionModel {
        SIMPLE,  ///< Motion model that assumes that the velocity is constant.
        INSTANT, ///< Motion model that takes the last measured velocity.
        SMOOTH,  ///< Motion model that models the velocity as a smooth evolving quantity parameterized by
                 ///< @ref smooth_mm_alpha_vel_.
        KALMAN   ///< Kalman motion model.
    };

    MotionModel motion_model_ = MotionModel::SIMPLE; ///< Type of motion model to use.

    // Smooth motion model related parameters.
    /// Smoothing parameter used in @ref MotionModel::SMOOTH
    double smooth_mm_alpha_vel_ = 0.001;
    bool smooth_mm_is_postponed = false;

    // Kalman motion model related parameters.
    /// @brief Defines the motion model used in @ref MotionModel::KALMAN.
    enum class KalmanModel {
        CONSTANT_VELOCITY,    ///< Motion model with a constant velocity.
        CONSTANT_ACCELERATION ///< Motion model with a constant acceleration.
    };

    /// @brief Defines the policy to determine measurement noise from data in @ref MotionModel::KALMAN.
    enum class KalmanPolicy {
        ADAPTIVE_NOISE,   ///< Policy that adapts the measurement noise based on the number of
                          ///< events associated to the tracker.
        MEASUREMENT_TRUST ///< Policy that keeps the noise model as defined in the provided parameters.
    };

    /// Standard deviation of the transition noise for the tracker's position in @ref MotionModel::KALMAN.
    double kalman_motion_model_pos_trans_std_ = 0.0001;
    /// Standard deviation of the transition noise for the tracker's velocity in @ref MotionModel::KALMAN.
    double kalman_motion_model_vel_trans_std_ = 0.05;
    /// Standard deviation of the transition noise for the tracker's acceleration in @ref MotionModel::KALMAN.
    double kalman_motion_model_acc_trans_std_ = 1e-9;
    /// Standard deviation of the observation noise for the tracker's position in @ref MotionModel::KALMAN.
    double kalman_motion_model_pos_obs_std_ = 100;
    /// Factor to multiply to noise variance at initialization in @ref MotionModel::KALMAN.
    double kalman_motion_model_init_factor_ = 1e12;
    /// Expected average events per pixel rate in events/us in @ref MotionModel::KALMAN.
    double kalman_motion_model_avg_events_per_pixel_ = 1;
    /// Minimal timestep at which Kalman filter is computed in @ref MotionModel::KALMAN.
    timestamp kalman_motion_model_min_dt_ = 1000;
    /// Policy used in @ref MotionModel::KALMAN.
    KalmanPolicy kalman_motion_model_policy_ = KalmanPolicy::ADAPTIVE_NOISE;
    /// Motion model used in @ref MotionModel::KALMAN.
    KalmanModel kalman_motion_model_motion_model_ = KalmanModel::CONSTANT_VELOCITY;

    /// @brief Defines the type of the tracker used by the @ref TrackingAlgorithm.
    enum class Tracker {
        ELLIPSE,  ///< Tracker is based on the event by event update of the tracker's pose and shape.
                  ///< The tracker is represented as a Gaussian, and the update is performed by weighting each event
                  ///< contribution with @ref EllipseUpdateFunction and updating the tracker's pose/size using
                  ///< @ref EllipseUpdateMethod.
        CLUSTERKF ///< Kalman tracker that uses the result of clustering as an observation state to predict the current
                  ///< state of the tracker. It considers the cluster's barycenter and size as measurements, and
                  ///< it estimates the tracker's position, velocity and size.
    };

    Tracker tracker_ = Tracker::CLUSTERKF; ///< Type of tracker to use.

    // Ellipse tracker related parameters.
    /// @brief Update function used in @ref Tracker::ELLIPSE
    enum class EllipseUpdateFunction {
        UNIFORM,            ///< Function outputs one if the radius is smaller than the parameter, and zero otherwise.
        GAUSSIAN,           ///< Function computes a Gaussian with mean equal to 1 and variance given in input.
        SIGNED_GAUSSIAN,    ///< Function computes a Gaussian multiplied by the radius minus the mean.
                            ///< This changes the sign of the function inside the radius and gives a negative update.
        TRUNCATED_GAUSSIAN  ///< Function computes a Gaussian if the radius is larger than the mean, otherwise
                            ///< returns zero.
    };

    /// @brief Method used in @ref Tracker::ELLIPSE to update the tracker
    enum class EllipseUpdateMethod {
        PER_EVENT,            ///< Method updates the tracker at each event
        ELLIPSE_FITTING,      ///< Method fits an ellipse on the associated cluster events and then updates the
                              ///< tracker's position event by event and the shape using the covariance of the
                              ///< associated cluster
        GAUSSIAN_FITTING,     ///< Method computes the mean and the covariance of the associated cluster events and then
                              ///< updates the tracker's position event by event and the shape using the covariance
                              ///< of the associated cluster
        ELLIPSE_FITTING_FULL, ///< Method fits an ellipse on the associated cluster events and then updates the tracker
                              ///< using a convex combination on the mean and covariance of the cluster without weight
        GAUSSIAN_FITTING_FULL ///< Method fits a Gaussian on the associated cluster events and then updates the tracker
                              ///< using a convex combination on the mean and covariance of the cluster without weight
    };

    double sigma_xx_                       = 5.;
    double sigma_yy_                       = 5.;
    double alpha_pos_                      = 0.1;  ///< Update rate for the tracker position in @ref Tracker::ELLIPSE
    double alpha_shape_                    = 0.04; ///< Update rate for the tracker shape in @ref Tracker::ELLIPSE
    EllipseUpdateFunction update_function_ = EllipseUpdateFunction::GAUSSIAN;       ///< Update function used in
                                                                                    ///< @ref Tracker::ELLIPSE
    double update_function_param_          = 100.;
    EllipseUpdateMethod update_method_     = EllipseUpdateMethod::GAUSSIAN_FITTING; ///< Update method used in
                                                                                    ///< @ref Tracker::ELLIPSE
    bool decompose_covariance_             = false;  ///< Flag specifying if to update the covariance decomposing it
                                                     ///< in eigenvalues and eigenvectors in @ref Tracker::ELLIPSE

    // Cluster KF related parameters.
    double cluster_kf_pos_var_      = 1200000000; ///< Variance of the tracker's position in @ref Tracker::CLUSTERKF
    double cluster_kf_vel_var_      = 32000;      ///< Variance of the tracker's velocity in @ref Tracker::CLUSTERKF
    double cluster_kf_acc_var_      = 0.8;        ///< Variance of the tracker's acceleration in @ref Tracker::CLUSTERKF
    double cluster_kf_size_var_     = 200000;     ///< Variance of the tracker's size in @ref Tracker::CLUSTERKF
    double cluster_kf_vel_size_var_ = 2;          ///< Variance of the tracker's size change rate in
                                                  ///< @ref Tracker::CLUSTERKF
    double cluster_kf_pos_obs_var_  = 200;        ///< Variance of the observed position in @ref Tracker::CLUSTERKF
    double cluster_kf_size_obs_var_ = 1e3;        ///< Variance of the observed size in @ref Tracker::CLUSTERKF

    // Tracker eraser related parameters.
    /// Delta ts (in us) for the position prediction.
    timestamp delta_ts_for_prediction_ = 5000;
    /// Time (in us) to wait without having updated a tracker before deleting it.
    timestamp ts_to_stop_            = 100000;
    /// File indicating the forbidden area
    std::string forbidden_area_file_ = "";

    // Not configurable from configfile (for now)
    double min_speed_ = 0;
    double max_speed_ = std::numeric_limits<float>::max();
};

inline std::istream &operator>>(std::istream &in, TrackingConfig::ClusterMaker &cm) {
    std::string token;
    in >> token;
    if (token == "SIMPLE_GRID")
        cm = TrackingConfig::ClusterMaker::SIMPLE_GRID;
    else if (token == "MEDOID_SHIFT")
        cm = TrackingConfig::ClusterMaker::MEDOID_SHIFT;
    return in;
}

inline std::istream &operator>>(std::istream &in, TrackingConfig::DataAssociation &da) {
    std::string token;
    in >> token;
    if (token == "NEAREST")
        da = TrackingConfig::DataAssociation::NEAREST;
    else if (token == "IOU")
        da = TrackingConfig::DataAssociation::IOU;
    return in;
}

inline std::istream &operator>>(std::istream &in, TrackingConfig::MotionModel &mm) {
    std::string token;
    in >> token;
    if (token == "SIMPLE")
        mm = TrackingConfig::MotionModel::SIMPLE;
    else if (token == "INSTANT")
        mm = TrackingConfig::MotionModel::INSTANT;
    else if (token == "SMOOTH")
        mm = TrackingConfig::MotionModel::SMOOTH;
    else if (token == "KALMAN")
        mm = TrackingConfig::MotionModel::KALMAN;
    return in;
}

inline std::istream &operator>>(std::istream &in, TrackingConfig::Tracker &mm) {
    std::string token;
    in >> token;
    if (token == "ELLIPSE")
        mm = TrackingConfig::Tracker::ELLIPSE;
    else if (token == "CLUSTERKF")
        mm = TrackingConfig::Tracker::CLUSTERKF;
    return in;
}

inline std::istream &operator>>(std::istream &in, TrackingConfig::KalmanPolicy &pol) {
    std::string token;
    in >> token;
    if (token == "ADAPTIVE_NOISE")
        pol = TrackingConfig::KalmanPolicy::ADAPTIVE_NOISE;
    else if (token == "MEASUREMENT_TRUST")
        pol = TrackingConfig::KalmanPolicy::MEASUREMENT_TRUST;
    return in;
}

inline std::istream &operator>>(std::istream &in, TrackingConfig::KalmanModel &mm) {
    std::string token;
    in >> token;
    if (token == "CONSTANT_VELOCITY")
        mm = TrackingConfig::KalmanModel::CONSTANT_VELOCITY;
    else if (token == "CONSTANT_ACCELERATION")
        mm = TrackingConfig::KalmanModel::CONSTANT_ACCELERATION;
    return in;
}

inline std::istream &operator>>(std::istream &in, TrackingConfig::EllipseUpdateFunction &uf) {
    std::string token;
    in >> token;
    if (token == "UNIFORM")
        uf = TrackingConfig::EllipseUpdateFunction::UNIFORM;
    else if (token == "GAUSSIAN")
        uf = TrackingConfig::EllipseUpdateFunction::GAUSSIAN;
    else if (token == "SIGNED_GAUSSIAN")
        uf = TrackingConfig::EllipseUpdateFunction::SIGNED_GAUSSIAN;
    else if (token == "TRUNCATED_GAUSSIAN")
        uf = TrackingConfig::EllipseUpdateFunction::TRUNCATED_GAUSSIAN;
    return in;
}

inline std::istream &operator>>(std::istream &in, TrackingConfig::EllipseUpdateMethod &um) {
    std::string token;
    in >> token;
    if (token == "PER_EVENT")
        um = TrackingConfig::EllipseUpdateMethod::PER_EVENT;
    else if (token == "ELLIPSE_FITTING")
        um = TrackingConfig::EllipseUpdateMethod::ELLIPSE_FITTING;
    else if (token == "GAUSSIAN_FITTING")
        um = TrackingConfig::EllipseUpdateMethod::GAUSSIAN_FITTING;
    else if (token == "ELLIPSE_FITTING_FULL")
        um = TrackingConfig::EllipseUpdateMethod::ELLIPSE_FITTING_FULL;
    else if (token == "GAUSSIAN_FITTING_FULL")
        um = TrackingConfig::EllipseUpdateMethod::GAUSSIAN_FITTING_FULL;
    return in;
}

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_TRACKING_ALGORITHM_CONFIG_H
