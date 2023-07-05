/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV3D_EDGE_LS_PROBLEM_H
#define METAVISION_SDK_CV3D_EDGE_LS_PROBLEM_H

#include "metavision/sdk/cv/utils/least_squares_problem_base.h"
#include "metavision/sdk/cv3d/utils/edge_data_association.h"

namespace Metavision {

/// @brief Cost function allowing estimating the camera's pose with respect to a 3D model
///
/// The pose is estimated by minimizing the orthogonal re-projection error between 3D points lying on the model's edges
/// and their associated matches in the undistorted normalized image plane.
class EdgeLSProblem : public LeastSquaresProblemBase<EdgeLSProblem, float, 6> {
public:
    friend class LeastSquaresProblemBase<EdgeLSProblem, float, 6>;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// @brief Constructor
    EdgeLSProblem();

    /// @brief Clears all the previously registered data associations
    void reset();

    /// @brief Sets the camera's pose to start the minimization from
    /// @param T_c_w Camera's pose
    void set_camera_pose(const Eigen::Matrix<Scalar, 4, 4> &T_c_w);

    /// @brief Adds a new data association to the cost function
    /// @param data_association Edge data association
    void add_data_association(const EdgeDataAssociation &data_association);

    /// @brief Returns the last pose computed by minimizing this cost function
    const Eigen::Matrix<Scalar, 4, 4> &get_pose() const;

private:
    bool get_updated_innovation_vector_impl(unsigned int iteration, Scalar &cost, CostJacobian *cost_jacobian,
                                            Hessian *cost_hessian);

    bool accumulate_parameter_increment_impl(const Parameters &parameter_increment);

    void notify_new_best_estimate_impl(unsigned int iteration);

    void notify_starting_minimization_impl();

    void notify_ending_minimization_impl();

    Parameters xi_;
    Parameters xi_best_;
    Jacobian jy_;
    Eigen::Matrix<Scalar, 4, 4> T_c_w_;
    EdgeDataAssociationVector edge_data_associations_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV3D_EDGE_LS_PROBLEM_H
