/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_LEAST_SQUARES_PROBLEM_BASE_H
#define METAVISION_SDK_CV_LEAST_SQUARES_PROBLEM_BASE_H

#include <Eigen/Core>

namespace Metavision {

/// @brief Base class for least-square problems to be solved using one of the generic solver implementations
/// @tparam Impl This class uses a Curiously Recursive Template Pattern design to avoid the cost of virtual methods. The
/// input template parameter is expected to inherit from this class and this class should be declared as friend of the
/// derived class.
/// @tparam FloatType Either float or double
/// @tparam D Dimension of the parameters vector to optimize
template<typename Impl, typename FloatType, int D>
class LeastSquaresProblemBase {
public:
    static_assert(std::is_floating_point<FloatType>::value, "Floating point type required");

    static constexpr int Dim = D;
    using Scalar             = FloatType;
    using Vec                = Eigen::Matrix<Scalar, Dim, 1>;
    using Jacobian           = Eigen::Matrix<Scalar, 1, Dim>;
    using CostJacobian       = Eigen::Matrix<Scalar, Dim, 1>;
    using Parameters         = Eigen::Matrix<Scalar, Dim, 1>;
    using Hessian            = Eigen::Matrix<Scalar, Dim, Dim>;

    /// @brief Computes the updated sum of squared residuals, innovation vector and hessian matrix
    ///
    /// The innovation vector and hessian matrix are only required if non-NULL pointers are provided as input arguments
    ///
    /// @param[in] iteration Integer indicating the current iteration
    /// @param[out] cost Reference of the value where the cost function evaluated at the current estimate should be
    /// stored
    /// @param[out] cost_jacobian Pointer to the DIMx1 vector where the Jacobian of the cost function evaluated at the
    /// current estimate should be stored. May be NULL, in which case the innovation vector is not needed.
    /// @param[out] cost_hessian Pointer to the DxD matrix where the hessian of the cost function evaluated at the
    /// current estimate should be stored. May be NULL, in which case the hessian matrix is not needed.
    /// @return true if the function succeeded, false if a fatal error occurred and the optimization algorithm should
    /// stop.
    /// @warning The behavior of this method must be implemented by implementing a get_updated_innovation_vector_impl
    /// method in the derived class.
    inline bool get_updated_innovation_vector(unsigned int iteration, Scalar &cost, CostJacobian *cost_jacobian,
                                              Hessian *cost_hessian);

    /// @brief Function to pass the estimated parameter increment and accumulate it in the parameter vector
    /// @param[in] parameter_increment Best parameter increment, estimated at the end of the current iteration, to be
    /// accumulated with the current parameter estimate
    /// @return true if the function succeeded, false if a fatal error occurred and the optimization algorithm should
    /// stop
    /// @warning The behavior of this method must be implemented by implementing an accumulate_parameter_increment_impl
    /// method in the derived class.
    inline bool accumulate_parameter_increment(const Parameters &parameter_increment);

    /// @brief Function to notify that the last parameter increment (given again as input argument) was not conclusive
    /// and should be canceled
    /// @param[in] parameter_increment Last accumulated parameter increment to be canceled
    /// @return true if the function succeeded, false if a fatal error occurred and the optimization algorithm should
    /// stop
    /// @note The behavior of this method can be redefined by implementing a cancel_parameter_increment_impl method in
    /// the derived class. Calls @ref accumulate_parameter_increment with @p -parameter_increment by default.
    inline bool cancel_parameter_increment(const Parameters &parameter_increment);

    /// @brief Function to notify that the parameter estimate obtained after the last increment is the best one so far
    /// @param iteration Integer indicating the current iteration
    /// @note The behavior of this method can be redefined by implementing a notify_new_best_estimate_impl method in the
    /// derived class. Does nothing by default.
    inline void notify_new_best_estimate(unsigned int iteration);

    /// @brief Notifies that the minimization is starting
    /// @note The behavior of this method can be redefined by implementing a notify_starting_minimization_impl method in
    /// the derived class. Does nothing by default.
    inline void notify_starting_minimization();

    /// @brief Notifies that the minimization is ending
    /// @note The behavior of this method can be redefined by implementing a notify_ending_minimization_impl method in
    /// the derived class. Does nothing by default.
    inline void notify_ending_minimization();

    /// @brief In case of minimization success this function notifies the Jacobian and Hessian matrices corresponding
    /// to the best estimate
    /// @param[in] jacobian Jacobian of the cost function evaluated at the best estimate
    /// @param[in] hessian Hessian of the cost function evaluated at the best estimate
    /// @note The behavior of this method can be redefined by implementing a notify_last_jacobian_and_hessian_impl
    /// method in the derived class. Does nothing by default.
    inline void notify_last_jacobian_and_hessian(const CostJacobian &jacobian, const Hessian &hessian);

private:
    Impl &child_cast();

    bool cancel_parameter_increment_impl(const Parameters &parameter_increment);
    void notify_new_best_estimate_impl(unsigned int iteration) {}
    void notify_starting_minimization_impl() {}
    void notify_ending_minimization_impl() {}
    void notify_last_jacobian_and_hessian_impl(const CostJacobian &jacobian, const Hessian &hessian) {}
};
} // namespace Metavision

#include "metavision/sdk/cv/utils/detail/least_squares_problem_base_impl.h"

#endif // METAVISION_SDK_CV_LEAST_SQUARES_PROBLEM_BASE_H