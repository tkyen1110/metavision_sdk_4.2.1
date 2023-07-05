/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_GAUSS_NEWTON_SOLVER_H
#define METAVISION_SDK_CV_GAUSS_NEWTON_SOLVER_H

#include <vector>
#include <ostream>

namespace Metavision {

template<typename Impl, typename Scalar, int D>
class LeastSquaresProblemBase;

namespace GaussNewton {

/// @brief Structure containing the parameters for deciding when to stop the minimization process
template<typename Scalar>
struct TerminationCriteria {
    static_assert(std::is_floating_point<Scalar>::value, "Floating point type required");

    /// @brief Default constructor
    TerminationCriteria() = default;

    unsigned int max_iterations       = 10; ///< Threshold on the maximum number of iterations
    Scalar min_increment_norm         = 0;  ///< Threshold on the minimum norm of the parameter increment
    Scalar min_cost_decrease_fraction = 0;  ///< Threshold on the minimum relative decrease of the cost
    Scalar max_cost_increase_fraction = 1;  ///< Threshold on the maximum relative increase of the cost

    /// @brief Initialization constructor
    /// @param max_iterations Threshold on the maximum number of iterations
    /// @param min_increment_norm Threshold on the minimum norm of the parameter increment
    /// @param min_cost_decrease_fraction Threshold on the minimum relative decrease of the cost
    /// @param max_cost_increase_fraction Threshold on the maximum relative increase of the cost
    TerminationCriteria(unsigned int max_iterations, Scalar min_increment_norm, Scalar min_cost_decrease_fraction,
                        Scalar max_cost_increase_fraction) :
        max_iterations(max_iterations),
        min_increment_norm(min_increment_norm),
        min_cost_decrease_fraction(min_cost_decrease_fraction),
        max_cost_increase_fraction(max_cost_increase_fraction) {}
};

/// @brief Structure containing information summarizing a minimization process
template<typename Scalar>
struct Report {
    static_assert(std::is_floating_point<Scalar>::value, "Floating point type required");

    /// @brief Enumeration of the possible states of the solver
    enum class State {
        MIN_INCREMENT_NORM_REACHED, ///< The threshold on the minimum norm of the parameter increment was reached
        ///< (i.e. the optimization converged).
        MIN_ERROR_DECREASE_FRACTION_REACHED, ///< The threshold on the minimum relative decrease of the cost was
        ///< reached (i.e. the optimization converged).
        MAX_ITERATIONS_REACHED,     ///< The threshold on the maximum number of iterations was reached.
        MAX_ERROR_INCREASE_REACHED, ///< The threshold on the maximum relative increase of the cost was reached (i.e.
        ///< the algorithm was unable to reduce the cost).
        ARITHMETIC_FAILURE, ///< An arithmetic failure happened in the course of the algorithm (most of the time,
        ///< it means the hessian matrix was singular).
        EXTERNAL_FAILURE ///< An external failure in the implementation of the AbstractLeastSquaresProblem<D,T>
        ///< happened in the course of the algorithm.
    };

    /// @brief Enumerate used to define the status of a minimization's iteration
    ///
    /// A minimization's iteration's status might consist in the combination of several of these values
    enum IterationStatus {
        JACOBIAN_VECTOR_ESTIMATION_FAILED  = 0x1,  ///< It was not possible to compute the cost or the Jacobian
        NORMAL_EQUATION_COMPUTATION_FAILED = 0x2,  ///< It was not possible to solve the Gauss-Newton's normal equation
        PARAMETER_ACCUMULATION_FAILED      = 0x4,  ///< It was not possible to accumulate the parameter increment
        COST_DECREASED                     = 0x8,  ///< The cost decreased during this iteration
        COST_INCREASED                     = 0x10, ///< The cost increased during this iteration
        LAST_COST_COMPUTATION_FAILED       = 0x20, ///< It was not possible to compute the cost at the last step of the
                                                   /// minimization process
        LAST_COST_DECREASED   = 0x40,              ///< The cost decreased at the last step of the minimization process
        LAST_COST_INCREASED   = 0x80,              ///< The cost increased at the last step of the minimization process
        LAST_COST_COMPUTATION = LAST_COST_COMPUTATION_FAILED | LAST_COST_DECREASED |
                                LAST_COST_INCREASED ///< Tells whether the iteration corresponds to the last cost
                                                    /// computation or not
    };

    /// @brief Structure containing information summarizing a minimization's iteration
    struct IterationReport {
        int status = 0;       ///< The iteration's status
        Scalar error;         ///< Cost computed during this iteration
        Scalar delta_error;   ///< Ratio (best_cost - new_cost) / best_cost
        Scalar sum_inc_norm;  ///< Infinity norm of the sum of all the parameters increments computed so far
        Scalar inc_norm;      ///< Infinity norm of the parameters increment computed during this iteration
        Scalar cost_inc_frac; ///< Ratio (new_cost / best_cost)
        bool is_result_nan;   ///< This flag is used in case of failure to determine if the parameters increment is
                              /// zero (false) or NAN (true)
    };

    /// @brief Constructor
    Report() {
        reset();
    }

    /// @brief Resets the report
    void reset() {
        state = State::MAX_ITERATIONS_REACHED;
        iter_reports.clear();
    }

    /// @brief Adds a new iteration report to the minimization process's report
    /// @return The newly added iteration report
    IterationReport &get_new_iter_report() {
        iter_reports.emplace_back();
        return iter_reports.back();
    }

    /// @brief Prints the minimization process's report
    /// @param stream Stream to print the report into
    /// @return The stream in which the report has been printed
    std::ostream &print(std::ostream &stream) const;

    /// @brief Override of the operator<< to print the minimization process's report into streams
    /// @param stream Stream to print the report into
    /// @param report Report to print into the input stream
    /// @return The stream in which the report has been printed
    friend std::ostream &operator<<(std::ostream &stream, const Report &report) {
        return report.print(stream);
    }

    State state;
    std::vector<IterationReport> iter_reports;
};

/// @brief Solves a non linear optimization problem using the Gauss-Newton method
/// @tparam Impl Type of the underlying least squares problem to solve
/// @tparam Scalar Either float or double
/// @tparam DIM Dimensions of the parameters to optimize
/// @tparam LeastSquaresProblemBase Base least squares problem type, the CRTP design is used here
/// @param problem Least squares problem to solve
/// @param report Final summary of the optimization problem
/// @param term_criteria Termination criteria used for deciding when to stop the minimization process
template<typename Impl, typename Scalar, int DIM,
         template<typename I, typename S, int D> typename LeastSquaresProblemBase>
void solve(LeastSquaresProblemBase<Impl, Scalar, DIM> &problem, Report<Scalar> &report,
           const TerminationCriteria<Scalar> &term_criteria = TerminationCriteria<Scalar>());

} // namespace GaussNewton
} // namespace Metavision

// Template implementation
#include "metavision/sdk/cv/utils/detail/gauss_newton_solver_impl.h"

#endif // METAVISION_SDK_CV_GAUSS_NEWTON_SOLVER_H
