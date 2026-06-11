#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

#include "walnuts/concepts.hpp"

namespace walnuts::detail {

#if defined(__has_attribute) && __has_attribute(always_inline)
#define WALNUTS_STRONG_INLINE [[gnu::always_inline]] inline
#else
#define WALNUTS_STRONG_INLINE inline
#endif

#ifdef __APPLE__
#include <pthread/qos.h>
WALNUTS_STRONG_INLINE void interactive_qos() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);  // best
}
WALNUTS_STRONG_INLINE void initiated_qos() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);  // next best
}
#else
WALNUTS_STRONG_INLINE void interactive_qos() {}
WALNUTS_STRONG_INLINE void initiated_qos() {}
#endif

/**
 * @brief A conservative constant destructive interference size.
 *
 * The std::hardware_destructive_interference_size is not universally supported
 * and can underreport when it is supported.  128 is safe for ARM and Intel
 * hardware.
 */
inline constexpr std::size_t FALSE_SHARING_GUARD_SIZE = 128;

/**
 * @brief Proposal update schemes for MCMC transitions.
 */
enum class Update {
  Barker,    /**< Use Barker's acceptance rule (proportional to density). */
  Metropolis /**< Use standard Metropolis acceptance rule */
};

/**
 * @brief Time direction of Hamiltonian simulation.
 */
enum class Direction {
  Backward, /**< Step backward in time. */
  Forward   /**< Step forward in time. */
};

/**
 * @brief A type definition for constructing `Direction::Backward` constants.
 */
using Backward_t = std::integral_constant<Direction, Direction::Backward>;

/**
 * @brief A type definition for constructing `Direction::Forward` constants.
 */
using Forward_t = std::integral_constant<Direction, Direction::Forward>;

/**
 * @brief A class encapsulating the randomizers needed for Hamiltonian Monte
 * Carlo.
 *
 * @tparam RNG The type of the base random number generator.
 */
template <std::uniform_random_bit_generator RNG>
class Random {
 public:
  /**
   * @brief Construct a randomizer with the specified base random number
   * generator.
   *
   * The base generator is held as a reference and used for all of the
   * generation. Thus it must be kept in scope as the instance constructed with
   * it is used.  The base generator may be shared with other applications.
   *
   * @param[in,out] rng The base random number generator.
   */
  explicit Random(RNG& rng) noexcept
      : rng_(rng), unif_(0.0, 1.0), binary_(0.5), normal_(0.0, 1.0) {}

  /**
   * @brief Return a number between 0 and 1 generated uniformly at random.
   *
   * The base random number generator is used to generate from a
   * `uniform([0, 1])` distribution.
   *
   * @return A number between 0 and 1 generated uniformly at random.
   */
  double uniform_real_01() { return unif_(rng_); }

  /**
   * @brief Return `true` or `false` uniformly at random.
   *
   * The base random number generator is used to generate from
   * a `uniform({0, 1})` distribution.
   *
   * @return A boolean value generated uniformly at random.
   */
  bool uniform_binary() { return binary_(rng_); }

  /**
   * @brief Return a vector of values generated according to a
   * standard normal distribution.
   *
   * The base random number generator is used to generate each
   * component independently from a `normal(0, 1)` distribution.
   *
   * @param[in] n The size of the vector generated.
   * @return An expression template for a standard normal vector.
   */
  auto standard_normal(Eigen::Index n) {
    return Eigen::VectorXd::NullaryExpr(
        n, [this](Eigen::Index /*i*/) { return this->normal_(rng_); });
  }

  /**
   * @brief Write a vector of random standard normal variables into the out
   * vector.
   *
   * The base random number generator is used to generate each
   * component independently from a `normal(0, 1)` distribution.
   *
   * @param[in] n The size of the vector generated.
   * @param[out] out The output vector.
   */
  void standard_normal(Eigen::Index n, Eigen::VectorXd& out) {
    out = standard_normal(n);
  }

  /**
   * @brief Return a reference to the base random number generator.
   *
   * @return The base random number generator.
   */
  RNG& rng() noexcept { return rng_; }

 private:
  /** The base random number generator reference. */
  RNG& rng_;  // not std::reference_wrapper to prevent copying

  /** The `uniform([0, 1])` random number generator. */
  std::uniform_real_distribution<double> unif_;

  /** The `uniform({0, 1})` random number generator. */
  std::bernoulli_distribution binary_;

  /** The `normal(0, 1)` random number generator. */
  std::normal_distribution<double> normal_;
};

/**
 * @brief Return the log of the sum of the exponentiated arguments.
 *
 * The mathematical definition is `log_sum_exp(x1, x2) = log(exp(x1) +
 * exp(x2))`.  The implementation is high precision and numerically stable.
 *
 * @param[in] x1 The first argument.
 * @param[in] x2 The second argument.
 * @return The log of the sum of the exponentiations of the arguments.
 */
inline double log_sum_exp(const double& x1, const double& x2) {
  auto m = std::fmax(x1, x2);
  if (std::isnan(x1) || std::isnan(x2)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (std::isinf(m) || std::isnan(x1 + x2)) {
    return std::fmax(x1, x2);
  }
  return m + std::log(std::exp(x1 - m) + std::exp(x2 - m));
}

/**
 * @brief Return the log of the sum of the components of the argument
 * exponentiated.
 *
 * The mathematical definition is `log_sum_exp(v) = log(sum(exp(x))`.
 * The implementation is high precision and numerically stable.
 *
 * @param[in] x The vector argument.
 * @return The log of the sum of the exponentiation of the vector's components.
 */
inline double log_sum_exp(const Eigen::VectorXd& x) {
  using std::log;
  if (x.size() == 0) {  // Eigen triggers assert on empty .maxCoeff()
    return -std::numeric_limits<double>::infinity();
  }
  double m = x.maxCoeff();
  if (std::isinf(m)) {
    return m;  // x[i] all -inf or all +inf; ow NaN
  }
  return m + log((x.array() - m).exp().sum());
}

/**
 * @brief Return the unnormalized log density of the specified momentum given
 * the specified inverse mass matrix diagonal.
 *
 * The unnormalized log density is the negative kinetic energy.
 *
 * The formula is `-0.5 * rho' .* inv_mass * rho`, which for diagonals works
 * out to `-0.5 * rho**2 * inv_mass` elementwise.
 *
 * @param[in] rho Vector of momenta.
 * @param[in] inv_mass_diag The diagonal of the diagonal inverse mass matrix.
 * @return The log density of the momentum.
 */
inline double logp_momentum(const Eigen::VectorXd& rho,
                            const Eigen::VectorXd& inv_mass_diag) {
  return -0.5 * (inv_mass_diag.array() * rho.array().square()).sum();
}

/**
 * @brief A wrapper for a log density and gradient function that traps
 * exceptions.
 *
 * @tparam F Type of underlying log density and gradient function.
 */
template <LogpGrad F>
class NoExceptLogpGrad {
 public:
  /**
   * @brief Construct a log density and gradient function from a base
   * log density and gradient function.
   *
   * The log density and gradient function will be stored as a
   * constant reference.
   *
   * @param[in] logp_grad The base log density and gradient function, called
   * back.
   */
  NoExceptLogpGrad(const F& logp_grad) : logp_grad_(std::cref(logp_grad)) {}

  /**
   * @brief Given the specified position, set the log density and
   * gradient.
   *
   * @param[in] x The position vector.
   * @param[out] logp The log density to set.
   * @param[out] grad The gradient to set.
   */
  void operator()(const Eigen::VectorXd& x, double& logp,
                  Eigen::VectorXd& grad) const noexcept {
    try {
      logp_grad_.get()(x, logp, grad);
    } catch (...) {
      // logp_grad failure equivalent to -inf log density
      // TODO: add logging for this kind of thing
      logp = -std::numeric_limits<double>::infinity();
      grad.setZero(x.size());
    }
  }

  /** The log density and gradient function. */
  const std::reference_wrapper<const F> logp_grad_;
};

/**
 * @brief Return the gradient of the log density at the specified position.
 *
 * @tparam F The type of the target log density/gradient function.
 * @param[in] logp_grad The target log density/gradient function.
 * @param[in] theta The position at which to evaluate the gradient.
 * @return The gradient of the log density at `theta`.
 */
inline Eigen::VectorXd grad(const LogpGrad auto& logp_grad,
                            const Eigen::VectorXd& theta) {
  Eigen::VectorXd g;
  double logp;
  logp_grad(theta, logp, g);
  return g;
}

/**
 * @brief Returns the L2 relative distance between the two vectors
 * scaled by the second vector.

 * The computation is `norm((a - b) / b)`.
 *
 * @param[in] a The test vector.
 * @param[in] b The baseline vector.
 * @return The relative difference
 */
inline double l2_rel_diff(const Eigen::VectorXd& a,
                          const Eigen::VectorXd& b) noexcept {
  return ((a - b).array() / b.array()).matrix().norm();
}

/**
 * @brief Return the sum of the sizes in the vector.
 *
 * @param[in] xs The vector to sum.
 * @return The sum.
 */
inline std::size_t sum(const std::vector<std::size_t>& xs) noexcept {
  return std::transform_reduce(xs.begin(), xs.end(), std::size_t(0),
                               std::plus<>{}, std::identity());
}

/**
 * @brief Return the bias-adjusted sample variance estimate.
 *
 * @param[in] xs The vector whose variance is required.
 * @return The variance.
 */
inline double variance(const Eigen::VectorXd& xs) noexcept {
  return (xs.array() - xs.mean()).square().sum() /
         static_cast<double>((xs.size() - 1));
}

}  // namespace walnuts::detail
