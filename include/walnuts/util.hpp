#pragma once

#include <Eigen/Dense>
#include <random>

namespace nuts {

/**
 * @brief The type of column vectors.
 * 
 * The type for dynamic Eigen column vectors with scalar type `S`.
 * @tparam S Type of scalars.
 */
template <typename S>
using Vec = Eigen::Matrix<S, Eigen::Dynamic, 1>;


/**
 * @brief The type of matrices.
 * 
 * The type for dynamic Eigen matrices with scalar type `S`.
 * @tparam S Type of scalars.
 */
template <typename S>
using Matrix = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
  

/**
 * @brief The type of integers.
 * 
 * Integers are signed and 64 bits.
 */
using Integer = std::int64_t;
  

/**
 * @brief Proposal update schemes for MCMC transitions.
 */
enum class Update {
  Barker,      /**< Use Barker's acceptance rule (proportional to density). */
  Metropolis   /**< Use standard Metropolis acceptance rule */
};

  
/**
 * @brief Time direction of Hamiltonian simulation.
 */
enum class Direction {
  Backward,    /**< Step backward in time. */
  Forward      /**< Step forward in time. */
};  

  
/**
 * @brief A class encapsulating the randomizers needed for Hamiltonian Monte Carlo.
 *
 * @tparam S The type of scalars.
 * @tparam RNG The type of the base random number generator.
 */
template <typename S, class RNG>
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
   * @param rng The base random number generator.
   */
  Random(RNG &rng)
      : rng_(rng), unif_(0.0, 1.0), binary_(0.5), normal_(0.0, 1.0)
  {}

  /**
   * @brief Return a number between 0 and 1 generated uniformly at random.
   *
   * The base random number generator is used to generate from a
   * `uniform([0, 1])` distribution.
   *
   * @return A number between 0 and 1 generated uniformly at random.
   */
  inline S uniform_real_01() { return unif_(rng_); }

  /**
   * @brief Return `true` or `false` uniformly at random.
   *
   * The base random number generator is used to generate from
   * a `uniform({0, 1})` distribution.
   *
   * @return A boolean value generated uniformly at random.
   */
  inline bool uniform_binary() { return binary_(rng_); }

  /**
   * @brief Return a vector of values generated according to a
   * standard normal distribution.
   *
   * The base random number generator is used to generate each
   * component independently from a `normal(0, 1)` distribution.
   *
   * @param n The size of the vector generated.
   * @return A vector generated according to a standard normal distribution.
   */
  inline Vec<S> standard_normal(Integer n) {
    return Vec<S>::NullaryExpr(n, [&](Integer) { return normal_(rng_); });
  }

 private:
  /** The base random number generator reference. */
  RNG &rng_;

  /** The `uniform([0, 1])` random number generator. */
  std::uniform_real_distribution<S> unif_;

  /** The `uniform({0, 1})` random number generator. */
  std::bernoulli_distribution binary_;

  /** The `normal(0, 1)` random number generator. */
  std::normal_distribution<S> normal_;
};


/**
 * @brief Return the log of the sum of the exponentiated arguments.
 *
 * The mathematical definition is `log_sum_exp(x1, x2) = log(exp(x1) +
 * exp(x2))`.  The implementation is high precision and numerically stable.
 *
 * @tparam S The type of scalars.
 * @param[in] x1 The first argument.
 * @param[in] x2 The second argument.
 * @return The log of the sum of the exponentiations of the arguments.
 */
template <typename S>
S log_sum_exp(const S &x1, const S &x2) {
  using std::fmax, std::log, std::exp;
  S m = fmax(x1, x2);
  return m + log(exp(x1 - m) + exp(x2 - m));
}


/**
 * @brief Return the log of the sum of the components of the argument
 * exponentiated.
 *
 * The mathematical definition is `log_sum_exp(v) = log(sum(exp(x))`.
 * The implementation is high precision and numerically stable.
 *
 * @tparam S The type of scalars.
 * @param[in] x The vector argument.
 * @return The log of the sum of the exponentiation of the vector's components.
 */
template <typename S>
S log_sum_exp(const Vec<S> &x) {
  using std::log;
  S m = x.maxCoeff();
  return m + log((x.array() - m).exp().sum());
}


/**
 * @brief Return the unnormalized log density of the specified momentum given
 * the specified inverse mass matrix. 
 *
 * The unnormalized log density is the negative kinetic energy.
 *
 * The formula is `-0.5 * rho' .* inv_mass * rho`.
 *
 * @tparam S Type of scalars.
 * @param[in] rho Vector of momenta.
 * @param[in] inv_mass The inverse mass matrix.
 * @return The log density of the momentum.
 */
template <typename S>
S logp_momentum(const Vec<S> &rho, const Vec<S> &inv_mass) {
  return -0.5 * rho.dot(inv_mass.cwiseProduct(rho));
}

  
/**
 * @brief Return a tuple of the arguments ordered by direction.

 * The arguments are forwarded as is the returned tuple.  If the
 * template argument `D` is `Direction::Forward`, then the tuple is
 * `(x1, x2)`; if `D` is `Direction::Backward`, the returned tuple
 * is `(x2, x1)`.
 * 
 * @tparam D The `Direction` in which to combine the arguments
 * (`Forward` or `Backward`).
 * @tparam T The type of the arguments.
 * @param[in] x1 The first argument.
 * @param[in] x2 The second argument.
 * @return The arguments ordered according to `D`.
 */
template <Direction D, typename T>
inline auto order_forward_backward(T &&x1, T &&x2) {
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
 * Equivalently, distance is measured in the Euclidean metric specified by the
 * mass matrix (i.e., the inverse of `inv_mass`).
 * 
 * If the spans ordered according to `D` are `(span_bk, span_fw)`, let
 * `theta_start` be the first position in `span_bk` and let `theta_end` be
 * the last position of `span_fw`. The U-turn condition will be satisfied if
 * ```
 * theta_start * delta < 0  OR   theta_end * delta < 0,
 * ```
 * where
 * ```
 * delta = `inv_mass .* (theta_end - theta_start)`.
 * ```
 *
 * @tparam D The direction in which to order the spans.
 * @tparam S The type of scalars.
 * @tparam U The type of spans, which must define begin and end positions.
 * @param[in] span_1 The first argument span.
 * @param[in] span_2 The second argument span.
 * @param[in] inv_mass The inverse mass matrix to determine distances.
 * @return `true` if there is a U-turn between the ends of the ordered spans.
 */  
template <Direction D, typename S, class U>
inline bool uturn(const U &span_1, const U &span_2, const Vec<S> &inv_mass) {
  auto &&[span_bk, span_fw] = order_forward_backward<D>(span_1, span_2);
  auto scaled_diff =
      (inv_mass.array() * (span_fw.theta_fw_ - span_fw.theta_bk_).array())
          .matrix();
  return span_fw.rho_fw_.dot(scaled_diff) < 0 ||
         span_bk.rho_bk_.dot(scaled_diff) < 0;
}

}  // namespace nuts
