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
 * Given a \f$D\f$-dimensional target density \f$p_\Theta\f$ and \f$D\f$-vector
 * \f$a\f$, let \f$\Phi = a^{-1} \odot \Theta\f$ so that \f$\Theta = a \odot
 * \Phi\f$. Because \f$a\f$ is constant, the change of variable rule yields
 *
 * \f[p_\Phi(\phi) \propto p_\Theta(a \cdot \phi)\f]
 *
 * and
 *
 * \f[\nabla p_\Phi(\phi) = a \odot \nabla p_\Theta(a \odot \phi)\f].
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

/**
 * @brief A handler filter to convert preconditioned varaibles back to
 * the original parameterization.
 */
template <typename H>
class PreconditionedHandler {
 public:

  /**
   * @brief Filter the specified handler with the specified preconditioner.
   *
   * Instances of this class hold the specified handler by reference,
   * so it must outlive the instance.
   *
   * @param[in] handler The handler to filter.
   * @param[in] a The preconditioning matrix.
   */
  PreconditionedHandler(H& handler, Eigen::VectorXd a)
      : handler_(handler), a_(std::move(a)) {}

  /**
   * @brief Handle a sample by passing the preconditioned version
   * to the nested handler.
   *
   * This method passes `a .* phi` to the handler provided to the constructor.
   *
   * @param phi Value to handle.
   * @param lp Log density value to handle.
   */
  void on_sample(const Eigen::VectorXd& phi, double lp) {
    Eigen::VectorXd theta = (a_.array() * phi.array()).matrix();
    handler_.get().on_sample(theta, lp);
  }
 private:
  std::reference_wrapper<H> handler_;
  Eigen::VectorXd a_;
};

  
} // namespace walnuts::detail
