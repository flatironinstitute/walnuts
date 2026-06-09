#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>

#include <Eigen/Dense>

#include <walnuts/concepts.hpp>

namespace walnuts::detail {

  /**
 * @brief A functor to linearly transform a log density and gradient function
 * for preconditioning.
 *
 * Given a $D$-dimensional target density $p_\Theta$ and $D$-vector
 * $a$, let $\Phi = a^{-1} \odot \Theta$ so that $\Theta = a \odot
 * \Phi$. Because $a$ is constant, the change of variable rule yields
 *
 * $p_\Phi(\phi) \propto p_\Theta(a \cdot \phi)$
 *
 * and
 *
 * $\nabla p_\Phi(\phi) = a \odot \nabla p_\Theta(a \odot \phi)$.
 *
 * @tparam F The type of the log density and gradient function to wrap.
 */
template <LogpGrad F>
class DiagPreconditionedLogpGrad {
 public:
  /**
   * @brief Construct a diagonally preconditioned log density and
   * gradient function.
   *
   * @param[in] f The original log density and gradient function.
   * @param[in] a The diagonal preconditioner.
   */
  DiagPreconditionedLogpGrad(F f, Eigen::VectorXd a)
    : f_(std::move(f)),
      a_(std::move(a)) {
  }

  /**
   * @brief Evaluate the preconditioned log density and gradient.
   *
   * @param[in] phi The variable to evaluate.
   * @param[out] logp The log density of `phi`.
   * @param[out] grad The gradient of the log density at `phi`.
   */
  void operator()(const Eigen::VectorXd& phi, double& logp,
		  Eigen::VectorXd& grad) const {
    Eigen::VectorXd theta = a_.array() * phi.array();
    f_(theta, logp, grad);
    grad.array() *= a_.array();
  }

  /**
   * @brief Return the diagonal of the diagonal preconditioning matrix.
   *
   * @return The diagonal preconditioner.
   */
  const Eigen::VectorXd& a() const noexcept { return a_; }

  /**
   * @brief Set the diagonal preconditioner to a new value.
   *
   * @param a The new diagonal preconditioner.
   */
  void set_a(const Eigen::VectorXd& a) {
    a_ = a;
  }
  
 private:
  F f_;
  Eigen::VectorXd a_;
  // mutable Eigen::VectorXd theta_, g_;  // reusable buffers for operator()
};

} // namespace walnuts::detail
