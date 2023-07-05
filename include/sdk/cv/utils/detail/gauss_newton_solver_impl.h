/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <iostream>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>

#include "metavision/sdk/base/utils/sdk_log.h"

#ifndef METAVISION_SDK_CV_GAUSS_NEWTON_SOLVER_IMPL_H
#define METAVISION_SDK_CV_GAUSS_NEWTON_SOLVER_IMPL_H

namespace Metavision {
namespace GaussNewton {

template<typename Scalar>
std::ostream &Report<Scalar>::print(std::ostream &stream) const {
    stream << "===== GaussNewtonSolver::solve() =====";

    using SizeType = typename std::vector<IterationReport>::size_type;
    for (SizeType i = 0; i < iter_reports.size(); ++i) {
        const auto &iter_report = iter_reports[i];

        if (iter_report.status & IterationStatus::LAST_COST_COMPUTATION) {
            if (state == State::MIN_INCREMENT_NORM_REACHED)
                stream << "\nReached the specified minimum increment norm (convergence)!\n";
            else
                stream << "\nReached the specified maximum number of iterations!\n";

            stream << "   end: ";

            if (iter_report.status & IterationStatus::LAST_COST_COMPUTATION_FAILED)
                stream << "\nFailed to compute last cost!\n";
            else {
                stream << "err=" << iter_report.error;
                if (iter_report.status & IterationStatus::LAST_COST_DECREASED)
                    stream << " (v)\n";
                else
                    stream << " (^)\n";
            }
        } else {
            stream << "\n" << std::right << std::setw(6) << i << std::setw(0) << std::left << ": ";
            if (iter_report.status & IterationStatus::JACOBIAN_VECTOR_ESTIMATION_FAILED) {
                stream << "Failed to compute updated jacobian vector!\n";
                continue;
            } else {
                stream << "err=" << iter_report.error;
                if (iter_report.status & IterationStatus::COST_DECREASED) {
                    stream << " (v), derr=" << iter_report.delta_error;
                    if (state == State::MIN_ERROR_DECREASE_FRACTION_REACHED) {
                        stream << "\nReached the specified minimum cost decrease fraction (convergence)!\n";
                        continue;
                    }
                } else if (iter_report.status & IterationStatus::COST_INCREASED) {
                    stream << " (^), derr=" << iter_report.delta_error;

                    if (state == State::MAX_ERROR_INCREASE_REACHED) {
                        stream << "\nReached the maximum allowed cost increase fraction (" << iter_report.cost_inc_frac
                               << ").\n";
                        continue;
                    }
                }
            }

            if (iter_report.status & IterationStatus::NORMAL_EQUATION_COMPUTATION_FAILED)
                stream << "\nFailed to solve the linear system (result is "
                       << (iter_report.is_result_nan ? "NAN" : "ZERO") << ")!\n";
            else if (iter_report.status & IterationStatus::PARAMETER_ACCUMULATION_FAILED)
                stream << "\nFailed to accumulate the parameter increment!\n";
            else {
                stream << ", |inc|/|suminc|=" << iter_report.inc_norm / iter_report.sum_inc_norm
                       << ", |inc|: " << iter_report.inc_norm << ", |suminc|: " << iter_report.sum_inc_norm;
            }
        }
    }

    stream << "===== ========================== =====\n";

    return stream;
}

template<typename Impl, typename Scalar, int DIM,
         template<typename I, typename S, int D> typename LeastSquaresProblemBase>
void solve(LeastSquaresProblemBase<Impl, Scalar, DIM> &problem, Report<Scalar> &report,
           const TerminationCriteria<Scalar> &term_criteria) {
    using LSProblem    = LeastSquaresProblemBase<Impl, Scalar, DIM>;
    using Report       = Report<Scalar>;
    using Vec          = typename LSProblem::Vec;
    using CostJacobian = typename LSProblem::CostJacobian;
    using Hessian      = typename LSProblem::Hessian;

    problem.notify_starting_minimization();

    bool has_unchecked_estimate = false;
    Vec sum_incs                = Vec::Zero();
    CostJacobian cost_jacobian  = CostJacobian::Zero();
    Hessian cost_hessian        = Hessian::Zero();

    report.reset();
    Scalar best_cost = std::numeric_limits<Scalar>::max(), new_cost;
    unsigned int iteration;
    for (iteration = 0; iteration < term_criteria.max_iterations; ++iteration) {
        auto &iter_report = report.get_new_iter_report();

        // Retrieve the updated hessian matrix and Jacobian vector
        cost_jacobian.setZero();
        cost_hessian.setZero();

        if (!problem.get_updated_innovation_vector(iteration, new_cost, &cost_jacobian, &cost_hessian)) {
            iter_report.status = Report::IterationStatus::JACOBIAN_VECTOR_ESTIMATION_FAILED;
            report.state       = Report::State::EXTERNAL_FAILURE;
            problem.notify_ending_minimization();
            return;
        }

        iter_report.error = new_cost;

        // Check if the cost increased or decreased
        has_unchecked_estimate  = false;
        iter_report.delta_error = (best_cost - new_cost) / best_cost;
        if (new_cost < best_cost) {
            iter_report.status = Report::IterationStatus::COST_DECREASED;
            if (iter_report.delta_error > Scalar(0) &&
                iter_report.delta_error < term_criteria.min_cost_decrease_fraction) {
                report.state = Report::State::MIN_ERROR_DECREASE_FRACTION_REACHED;
            }
            best_cost = new_cost;

            problem.notify_new_best_estimate(iteration);
            if (report.state == Report::State::MIN_ERROR_DECREASE_FRACTION_REACHED)
                break;
        } else {
            iter_report.status = Report::IterationStatus::COST_INCREASED;

            if (new_cost > term_criteria.max_cost_increase_fraction * best_cost) {
                iter_report.cost_inc_frac = new_cost / best_cost;
                report.state              = Report::State::MAX_ERROR_INCREASE_REACHED;
                problem.notify_ending_minimization();
                return;
            }
        }

        // Solve the normal equation using the computed hessian matrix and Jacobian vector
        const Vec inc = -cost_hessian.ldlt().solve(cost_jacobian);
        if (std::isnan(inc(0)) || (inc.squaredNorm() == 0 && cost_jacobian.squaredNorm() > 0)) {
            iter_report.status |= Report::IterationStatus::NORMAL_EQUATION_COMPUTATION_FAILED;
            iter_report.is_result_nan = std::isnan(inc(0));
            report.state              = Report::State::ARITHMETIC_FAILURE;
            problem.notify_ending_minimization();
            return;
        }
        if (!problem.accumulate_parameter_increment(inc)) {
            iter_report.status |= Report::IterationStatus::PARAMETER_ACCUMULATION_FAILED;
            report.state = Report::State::EXTERNAL_FAILURE;
            problem.notify_ending_minimization();
            return;
        }
        has_unchecked_estimate = true;

        // Check for convergence
        sum_incs += inc;
        iter_report.sum_inc_norm = sum_incs.template lpNorm<Eigen::Infinity>();
        iter_report.inc_norm     = inc.template lpNorm<Eigen::Infinity>();
        if (iter_report.inc_norm < term_criteria.min_increment_norm * iter_report.sum_inc_norm) {
            report.state = Report::State::MIN_INCREMENT_NORM_REACHED;
            break;
        }
    }

    // At this point, we may have accumulated a new parameter increment and not checked yet whether the new estimate is
    // the new best or not, so check this now
    if (has_unchecked_estimate) {
        auto &iter_report              = report.get_new_iter_report();
        CostJacobian new_cost_jacobian = CostJacobian::Zero();
        Hessian new_cost_hessian       = Hessian::Zero();

        if (!problem.get_updated_innovation_vector(iteration, new_cost, &new_cost_jacobian, &new_cost_hessian)) {
            iter_report.status = Report::IterationStatus::LAST_COST_COMPUTATION_FAILED;
        } else {
            iter_report.error = new_cost;
            if (new_cost < best_cost) {
                iter_report.status = Report::IterationStatus::LAST_COST_DECREASED;
                cost_jacobian      = new_cost_jacobian;
                cost_hessian       = new_cost_hessian;

                problem.notify_new_best_estimate(iteration);
            } else
                iter_report.status = Report::IterationStatus::LAST_COST_INCREASED;
        }
    }

    problem.notify_last_jacobian_and_hessian(cost_jacobian, cost_hessian);
    problem.notify_ending_minimization();
}
} // namespace GaussNewton
} // namespace Metavision

#endif // METAVISION_SDK_CV_GAUSS_NEWTON_SOLVER_IMPL_H
