/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_LEAST_SQUARES_PROBLEM_BASE_IMPL_H
#define METAVISION_SDK_CV_LEAST_SQUARES_PROBLEM_BASE_IMPL_H

namespace Metavision {
template<typename Impl, typename FloatType, int D>
inline bool LeastSquaresProblemBase<Impl, FloatType, D>::get_updated_innovation_vector(unsigned int iteration,
                                                                                       Scalar &cost,
                                                                                       CostJacobian *cost_jacobian,
                                                                                       Hessian *cost_hessian) {
    return child_cast().get_updated_innovation_vector_impl(iteration, cost, cost_jacobian, cost_hessian);
}

template<typename Impl, typename FloatType, int D>
inline bool
    LeastSquaresProblemBase<Impl, FloatType, D>::accumulate_parameter_increment(const Parameters &parameter_increment) {
    return child_cast().accumulate_parameter_increment_impl(parameter_increment);
}

template<typename Impl, typename FloatType, int D>
inline bool
    LeastSquaresProblemBase<Impl, FloatType, D>::cancel_parameter_increment(const Parameters &parameter_increment) {
    return child_cast().cancel_parameter_increment_impl(parameter_increment);
}

template<typename Impl, typename FloatType, int D>
inline void LeastSquaresProblemBase<Impl, FloatType, D>::notify_new_best_estimate(unsigned int iteration) {
    child_cast().notify_new_best_estimate_impl(iteration);
}

template<typename Impl, typename FloatType, int D>
inline void LeastSquaresProblemBase<Impl, FloatType, D>::notify_starting_minimization() {
    child_cast().notify_starting_minimization_impl();
}

template<typename Impl, typename FloatType, int D>
inline void LeastSquaresProblemBase<Impl, FloatType, D>::notify_ending_minimization() {
    child_cast().notify_ending_minimization_impl();
}

template<typename Impl, typename FloatType, int D>
inline void LeastSquaresProblemBase<Impl, FloatType, D>::notify_last_jacobian_and_hessian(const CostJacobian &jacobian,
                                                                                          const Hessian &hessian) {
    child_cast().notify_last_jacobian_and_hessian_impl(jacobian, hessian);
}

template<typename Impl, typename FloatType, int D>
Impl &LeastSquaresProblemBase<Impl, FloatType, D>::child_cast() {
    return static_cast<Impl &>(*this);
}

template<typename Impl, typename FloatType, int D>
inline bool LeastSquaresProblemBase<Impl, FloatType, D>::cancel_parameter_increment_impl(
    const Parameters &parameter_increment) {
    return child_cast().accumulate_parameter_increment_impl(-parameter_increment);
}

} // namespace Metavision

#endif // METAVISION_SDK_CV_LEAST_SQUARES_PROBLEM_BASE_IMPL_H
