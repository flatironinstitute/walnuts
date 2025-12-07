#pragma once

#include <Eigen/Dense>

#include "util.hpp"

namespace nuts {

/**
 * @brief An accumulator estimating discounted means and variances online.
 *
 * The `observe()` method receives vector value updates and maintains a
 * running estimate of discounted means and variances.  Historical counts
 * are discounted by the multiplying by the discount factor before each
 * new observation is added (with a count of one).  1 does no discounting
 * and 0 completely forgets the past.
 *
 * The implementation uses a weighted variant of Welford's algorithm that
 * discounts past observations.  It requires a constant memory of size
 * proportional to the dimensionality of the observed vectors (i.e.,
 * O(`dim`)).  Each of its methods runs in time proportional to the size
 * of the update vectors (i.e., O(`dim`)).  Arithmetic is stable
 * following the original Welford accumulator, to which it reduces
 * when `discount_factor = 1`.
 *
 * After initialization and updating with `N` vectors
 * `y[0], ..., y[N - 1]`, the weight for vector `y[n]` is
 * ```
 * weight[n] = discount_factor^(N - n - 1).
 * ```
 *
 * The discounted mean is calculated in the usual way for weighted
 * averages,
 * ```
 * mean = sum(y .* weight) / sum(weight),
 * ```
 * where `.*` is elementwise product.
 *
 *
 * The discounted mean is then used to estimate the discounted variance,
 *
 * ```
 * var = sum(weight .* (y - mean)^2) / sum(weight).
 * ```
 *
 * @tparam S The type of scalars.
 */
template <typename S>
class OnlineMoments {
 public:
  /**
   * @brief Construct a default online estimator of size zero.
   */
  OnlineMoments()
    : discount_factor_(0.98),  // dummy valid inits
      weight_(0),
      mean_(Vec<S>::Zero(0)),
      sum_sq_dev_(Vec<S>::Zero(0)) { }

  /**
   * @brief Construct online moments with a given discount factor and size.
   *
   * @param[in] discount_factor The past discount factor (between 0 and 1,
   * inclusive).
   * @param[in] dims The number of dimensions.
   * @throw std::invalid_argument If `discount_factor` is not in [0, 1].
   */
  OnlineMoments(double discount_factor, std::size_t dims)
    : discount_factor_(discount_factor),
      weight_(0),
      mean_(Vec<S>::Zero(static_cast<long>(dims))),
      sum_sq_dev_(Vec<S>::Zero(static_cast<long>(dims))) {
  }
    

  
  /**
   * @brief Construct an online estimator of moments with the
   * specified discount factor and initialization.
   *
   * The initialization specifies the initial mean and the initial
   * variance, assigning them a weight that is interpreted as if the
   * initial mean and variance were the result of a count of
   * `init_weight` observations.
   *
   * @param[in] discount_factor The past discount factor (between 0 and 1,
   * inclusive).
   * @param[in] init_weight Weight (in number of draws) of initial mean
   * and variance (positive).
   * @param[in] init_mean Initial mean.
   * @param[in] init_variance Initial variance.
   * @throw std::invalid_argument If `discount_factor` is not in (0, 1).
   * @throw std::invalid_argument If the initial weight is not finite and
   * positive.
   * @throw std::invalid_argument If the initial mean and variance are not
   * the same size.
   */
  OnlineMoments(S discount_factor, S init_weight, const Vec<S>& init_mean,
                const Vec<S>& init_variance)
      : discount_factor_(discount_factor),
        weight_(init_weight),
        mean_(init_mean),
        sum_sq_dev_(init_weight * init_variance) {
    validate_probability_inclusive(discount_factor, "discount_factor");
    validate_positive(init_weight, "init_weight");
    validate_same_size(init_mean, init_variance, "init_mean", "init_variance");
  }

  /**
   * @brief Set the discount factor for previous observations to the specified
   * value.
   *
   * @param discount_factor The discount factor.
   * @throw std::invalid_argument If the discount factor is not in (0, 1).
   */
  inline void set_discount_factor(S discount_factor) {
    validate_probability_inclusive(discount_factor, "set_discount_factor(discount_factror)");
    discount_factor_ = discount_factor;
  }

  /**
   * @brief Update this accumulator with the specified observation.
   *
   * The observed value `y` is assigned a weight (or count) of 1, and
   * the weights of the past observations are discounted by the discount
   * factor.
   *
   * @tparam Derived The type of matrix underlying the observation.
   * @param y The observed vector.
   * @pre y.size() == mean().size()
   */
  template <typename Derived>
  inline void observe(const Eigen::MatrixBase<Derived>& y) {
    const Vec<S> delta = y - mean_;
    weight_ = discount_factor_ * weight_ + 1;
    mean_ += delta / weight_;
    sum_sq_dev_ =
        discount_factor_ * sum_sq_dev_ + delta.cwiseProduct(y - mean_);
  }

  /**
   * @brief Set the discount factor, then update with the specified observation.
   *
   * This is a convenience method to call `set_discount_factor(discount_factor)`
   * and `observe(y)`.
   *
   * @tparam Derived The type of matrix underlying the observation.
   * @param discount_factor The discount factor.
   * @param y The observed vector.
   * @pre discount_factor > 0 && discount_factor <= 1
   * @pre y.size() == mean().size()
   */
  template <typename Derived>
  inline void discount_observe(double discount_factor,
                               const Eigen::MatrixBase<Derived>& y) {
    set_discount_factor(discount_factor);
    observe(y);
  }

  /**
   * @brief Return the estimate of the mean.
   *
   * @return The mean estimate.
   */
  inline const Vec<S>& mean() const noexcept { return mean_; }

  /**
   * @brief Return the estimate of the variance.
   *
   * @return The variance estimate.
   */
  inline Vec<S> variance() const {
    if (weight_ > 0) {
      return sum_sq_dev_ / weight_;
    }
    return Vec<S>::Ones(mean_.size());
  }

 private:
  /** The discount factor applied to the weights of previous observations. */
  S discount_factor_;

  /** The combined weight of all previous observations. */
  S weight_;

  /** The current mean estimate */
  Vec<S> mean_;

  /** The sum of weighted squared deviations from the mean. */
  Vec<S> sum_sq_dev_;
};

}  // namespace nuts
