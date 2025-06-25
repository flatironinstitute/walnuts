#pragma once

#include <Eigen/Dense>

namespace nuts {

/**
 * The `DiscountedOnlineMoments` accumulator for means and variances.
 * The `update()` method receives vector value updates and maintains a
 * running estimate of discounted means and variances.  The discount
 * factor is between 0 and 1, with larger values doing less
 * discounting, with 1 doing no discounting and 0 doing full
 * discounting.

 * This class requires a constant memory of size proportional to the
 * dimensionality of the update vectors (i.e., O(dim)).  Each of its
 * methods runs in time proportional to the size of the update vectors
 * (i.e., O(dim)).  Arithmetic is stable following the original Welford
 * accumulator, to which it reduces when `alpha = 1`.
 *
 * After initialization and updating with `N` vectors `y[0], ..., y[N
 - 1]`, the weight for vector `y[n]` is

 * ```
 * weight[n] = alpha^(N - n - 1).
 * ```
 *
 * The discounted mean is calculated in the usual way for weighted
 * averages,
 *
 * ```
 * mean = sum(y .* weight) / sum(weight).
 * ```
 *
 * The discounted mean is then used to calculate discounted variance,
 *
 * ```
 * var = sum(weight .* (y - mean) .* (y - mean)) / sum(weight).
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
   * Construct an online estimator of moments of the given
   * dimensionality that discounts the past by a factor of `alpha`
   * before each observation.  Setting `alpha` to 1 does no
   * discounting.
   *
   * @param alpha The past discount factor (between 0 and 1, inclusive).
   * @param dim Number of dimensions in observation vectors (non-negative
   *     integer).
   */
  DiscountedOnlineMoments(S alpha,
                          int dim)
      : alpha_(alpha),
        weight_(0),
        mu_(Vec::Zero(dim)),
        s_(Vec::Zero(dim)) { }

  /**
   * Construct an online estimamator of moments with the specified
   * discount factor and initialized at the specified mean and
   * variance with a weight equal to the specified pseudocount.
   *
   * @param alpha The past discount factor (between 0 and 1,
   * inclusive).  
   * @param init_weight Weight (in number of draws) of initial mean
   * and variance (positive).
   * @param init_mean Initial mean.
   * @param init_variance Initial variance.
   */
  DiscountedOnlineMoments(S alpha,
                          S init_weight,
                          const Vec& init_mean,
                          const Vec& init_variance)
      : alpha_(alpha),
        weight_(init_weight),
        mu_(init_mean),
        s_(init_weight * init_variance) { }

  /**
   * Update the state of this accumulator by observing the specified vector.
   *
   * @param y The observed vector.
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
    if (weight_ > 0) {
      return s_ / weight_;
    }
    return Vec::Zero(mu_.size());
  }

  /**
   * @brief Set the discount factor for previous observations to the specified
   * value.
   */
  void set_alpha(S alpha) {
    alpha_ = alpha;
  }

 private:
  /** The discount factor applied to all previous observations. */
  S alpha_;

  /** The combined weight in sample size of all previous observations. */
  S weight_;

  /** The current mean estimate */
  Vec mu_;

  /** The current variance estimate scaled by the weight. */
  Vec s_;
};

}  // namespace nuts
