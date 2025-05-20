#ifndef WALNUTS_WELFORD_HPP
#define WALNUTS_WELFORD_HPP

#include <Eigen/Dense>

namespace walnuts {

/**
 * The `DiscountedOnlineMoments` has an `update()` method that
 * receives vectors and updates its constant internal memory in order
 * to provide estimates of means and variances at any point.  The past
 * is optionally discounted to upweight more recent observations.
 *
 * In addition to the discount factor `alpha`, the algorithm maintains
 * a scalar `weight` and two vectors, `mu` and `s`, which store the
 * sufficient statistics required to estimate mean and variance.
 * The initialization is all zeros,
 *
 * ```
 * weight = 0;  mu = 0;  s = 0
 * ```
 *
 * Updates for a new observation y are given by
 *
 * ```
 * diff = y - mu_;
 * weight_ = alpha * weight + 1
 * mu_ = mu + delta / weight
 * s = alpha * s + delta * (y - mu)
 * ```
 *
 * At any given point in time, the current estimates of mean and
 * variance are given by
 *
 * ```
 * mean = mu
 * variance = s / weight
 * ```
 *
 * @tparam S The type of scalars.
 */
template <typename S>
class DiscountedOnlineMoments {
 public:
  /**
   * The type of vectors with scalar type `S`.
   */
  using Vec = Eigen::Matrix<S, Eigen::Dynamic, 1>;

  /**
   * Construct an online estimator of moments of the given dimensionality that
   * discounts the past by a factor of `alpha in (0, 1]` before each
   * observation.  Setting `alpha` to 1 does no discounting.
   *
   * @param alpha The discount factor.
   * @param dim Number of dimensions in observation vectors.
   */
  DiscountedOnlineMoments(S alpha, int dim)
      : alpha_(alpha), weight_(0), mu_(Vec::Zero(dim)), s_(Vec::Zero(dim)) {}

  /**
   * Update the state of this accumulator by observing the specified vector.
   *
   * @param y An observation vector.
   */
  void update(const Vec& y) {
    const Vec delta = y - mu_;
    weight_ = alpha_ * weight_ + 1;
    mu_ += delta / weight_;
    s_ = alpha_ * s_ + delta.cwiseProduct(y - mu_);
  }

  /**
   * Return the estimate of the mean.
   *
   * @return The mean estimate.
   */
  const Vec& mean() const { return mu_; }

  /**
   * Return the estimate of the variance.
   *
   * @return The variance estimate.
   */
  Vec variance() const {
    if (weight_ > 0) return s_ / weight_;
    return Vec::Zero(mu_.size());
  }

 private:
  S alpha_;
  S weight_;
  Vec mu_;
  Vec s_;
};

}  // namespace walnuts

#endif  // WALNUTS_WELFORD_HPP
