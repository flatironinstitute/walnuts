#pragma once

#include <random>
#include <ranges>
#include <type_traits>

#include <Eigen/Dense>

namespace walnuts {

#if defined(__has_attribute) && __has_attribute(always_inline)
#define WALNUTS_STRONG_INLINE [[gnu::always_inline]] inline
#else
#define WALNUTS_STRONG_INLINE inline
#endif

#ifdef __APPLE__
#include <pthread.h>
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

using Backward_t = std::integral_constant<Direction, Direction::Backward>;
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
  explicit Random(RNG& rng)
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
   * @brief Write a vector of random standard normal variables into the out
   * vector.
   *
   * The base random number generator is used to generate each
   * component independently from a `normal(0, 1)` distribution.
   *
   * @param[in] n The size of the vector generated.
   * @param[out] out The output vector.
   */
  void standard_normal(std::size_t n, Eigen::VectorXd& out) {
    out.resize(static_cast<Eigen::Index>(n));
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n); ++i) {
      out[i] = normal_(rng_);
    }
  }

  /**
   * @brief Return a vector of values generated according to a
   * standard normal distribution.
   *
   * The base random number generator is used to generate each
   * component independently from a `normal(0, 1)` distribution.
   *
   * @param[in] n The size of the vector generated.
   * @return A vector generated according to a standard normal distribution.
   */
  Eigen::VectorXd standard_normal(std::size_t n) {
    Eigen::VectorXd out;
    standard_normal(n, out);
    return out;
  }

  Eigen::VectorXd standard_normal_cwise_product(const Eigen::VectorXd& v) {
    Eigen::VectorXd out(v.size());
    for (Eigen::Index i = 0; i < v.size(); ++i) {
      out[i] = v[i] * normal_(rng_);
    }
    return out;
  }

 private:
  /** The base random number generator reference. */
  RNG& rng_;

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
  using std::fmax, std::log, std::exp;
  double m = fmax(x1, x2);
  if (std::isinf(m) && m < 0) {
    return m;  // x1 = x2 = -inf
  }
  return m + log(exp(x1 - m) + exp(x2 - m));
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
  double m = x.maxCoeff();
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
  // equiv, but with temporaries: -0.5 *
  // rho.dot(inv_mass_diag.cwiseProduct(rho));
  return -0.5 * (inv_mass_diag.array() * rho.array().square()).sum();
}

/**
 * @brief Return a tuple of the arguments ordered by direction.

 * The arguments are forwarded as is the returned tuple and returned
 * by reference, so function arguments must stay in scope.  If the
 * template argument `D` is `Direction::Forward`, then the tuple is
 * `(x1, x2)`; if `D` is `Direction::Backward`, the returned tuple is
 * `(x2, x1)`.
 *
 * @tparam D The `Direction` in which to combine the arguments
 * (`Forward` or `Backward`).
 * @tparam T The type of the arguments.
 * @param[in] x1 The first argument.
 * @param[in] x2 The second argument.
 * @return The arguments ordered according to `D`.
 */
template <Direction D, typename T>
inline auto order_forward_backward(T&& x1, T&& x2) {
  if constexpr (D == Direction::Forward) {
    return std::forward_as_tuple(std::forward<T>(x1), std::forward<T>(x2));
  } else {  // Direction::Backward
    return std::forward_as_tuple(std::forward<T>(x2), std::forward<T>(x1));
  }
}

/**
 * @brief Return `true` if the two spans ordered as specified form a
 * U-turn in the metric determined by the inverse mass matrix.
 *
 * For computing U-turns, the squared distance between two vectors `x` and `y`
 * is defined by the Mahalanobis distance,
 * ```
 * d(x, y)**2 = (x - y)' * inv_mass * (x - y).
 * ```
 * Equivalently, distance is measured in the Euclidean metric with metric
 * tensor given by the mass matrix.
 *
 * If the spans ordered according to `D` are `(span_bk, span_fw)`, let
 * `theta_start` be the first position in `span_bk` and let `theta_end` be
 * the last position of `span_fw`. The U-turn condition will be satisfied if
 * ```
 * theta_start * delta < 0  OR   theta_end * delta < 0,
 * ```
 * where
 * ```
 * delta = inv_mass .* (theta_end - theta_start).
 * ```
 *
 * @tparam D The direction in which to order the spans.
 * @tparam U The type of spans, which must define begin and end positions.
 * @param[in] span1 The first argument span.
 * @param[in] span2 The second argument span.
 * @param[in] inv_mass The inverse mass matrix to determine distances.
 * @return `true` if there is a U-turn between the ends of the ordered spans.
 */
template <Direction D, class U>
inline bool uturn(const U& span1, const U& span2,
                  const Eigen::VectorXd& inv_mass) {
  auto&& [span_bk, span_fw] = order_forward_backward<D>(span1, span2);
  auto scaled_diff =
      (inv_mass.array() * (span_fw.theta_fw_ - span_bk.theta_bk_).array())
          .matrix();
  return span_fw.rho_fw_.dot(scaled_diff) < 0 ||
         span_bk.rho_bk_.dot(scaled_diff) < 0;
}

/**
 * @brief Concept for a log density and gradient function.
 *
 * A type `F` satisfies `LogpGrad` if an object of type `const F&` can be
 * called with arguments `(const Eigen::VectorXd&, double&, Eigen::VectorXd&)`
 * and the call returns `void`. The first argument is the position at which
 * to evaluate, and the second and third are output parameters set to the
 * log density and its gradient, respectively.
 *
 * The callable is permitted to throw exceptions; see `ExceptionFreeLogpGrad`
 * for the noexcept variant.
 *
 * @tparam F The callable type to constrain.
 */  
template <typename F>
concept LogpGrad = requires(const F& f, const Eigen::VectorXd& x,
                            double& logp, Eigen::VectorXd& grad) {
    { f(x, logp, grad) } -> std::same_as<void>;
};
  
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
template <LogpGrad F>
Eigen::VectorXd grad(const F& logp_grad, const Eigen::VectorXd& theta) {
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
static double l2_rel_diff(const Eigen::VectorXd& a,
                          const Eigen::VectorXd& b) noexcept {
  return ((a - b).array() / b.array()).matrix().norm();
}

}  // namespace walnuts
