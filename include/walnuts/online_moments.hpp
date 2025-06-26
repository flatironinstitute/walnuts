#pragma once

#include <stdexcept>

#include <Eigen/Dense>

namespace nuts {

/**
 * @brief An accumulator maintaining discounted means and variances online.
 *
 * The `observe()` method receives vector value updates and maintains a
 * running estimate of discounted means and variances.  The discount
 * factor is between 0 and 1, with larger values doing less
 * discounting, with 1 doing no discounting and 0 doing full
 * discounting.
 *
 * The implementation uses a weighted form of Welford's algorithm that
 * discounts past observations.  It requires a constant memory of size
 * proportional to the dimensionality of the update vectors (i.e.,
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
class OnlineMoments {
  using Integer = long;
 public:
  /**
   * The type of vectors with scalar type `S`.
   */
  using VecS = Eigen::Matrix<S, Eigen::Dynamic, 1>;

  /**
   * @brief Construct an online estimator of moments of the given
   * dimensionality that discounts the past by a factor of
   * `discount_factor` before each observation.  Setting
   * `discount_factor` to 1 does no discounting.
   *
   * @param discount_factor The past discount factor (floating point in [0, 1]).
   * @param dim Number of dimensions in observation vectors (non-negative
   *     integer).
   * @throw std::invalid_argument If `discount_factor` is not in [0, 1].
   * @throw std::invalid_argument If `dim` is negative.
   */
  OnlineMoments(S discount_factor,
		Integer dim)
      : discount_factor_(discount_factor),
        weight_(0),
        mean_(VecS::Zero(dim)),
        sum_sq_dev_(VecS::Zero(dim)) {
    if (dim < 0) {
      throw std::invalid_argument("DiscountedOnlineMoments: "
				  "dim must be non-negative");
    }
    if (!(discount_factor >= 0 && discount_factor <= 1)) {
      throw std::invalid_argument("DiscountedOnlineMoments: "
				  "discount_factor must be in [0, 1]");
    }
  }

  /**
   * @brief Construct an online estimamator of moments with the
   * specified discount factor and initialized at the specified mean
   * and variance with a weight equal to the specified pseudocount.
   *
   * @param discount_factor The past discount factor (between 0 and 1,
   * inclusive).  
   * @param init_weight Weight (in number of draws) of initial mean
   * and variance (positive).
   * @param init_mean Initial mean.
   * @param init_variance Initial variance.
   * @throw std::invalid_argument If `discount_factor` is not in [0, 1].
   * @throw std::invalid_argument If the initial weight is not finite and
   * positive.
   * @throw std::invalid_argument If the initial mean and variance are not
   * the same size.
   */
  OnlineMoments(S discount_factor,
		S init_weight,
		const VecS& init_mean,
		const VecS& init_variance)
      : discount_factor_(discount_factor),
        weight_(init_weight),
        mean_(init_mean),
        sum_sq_dev_(init_weight * init_variance) {
    if (!(discount_factor >= 0 && discount_factor <= 1)) {
      throw std::invalid_argument("Discount factor must be in [0, 1].");
    }
    if (!(init_weight > 0) || std::isinf(init_weight)) {
      throw std::invalid_argument("Initial weight must be finite and positive.");
    }
    if (init_mean.size() != init_variance.size()) {
      throw std::invalid_argument("Initial mean and variance must be same size.");
    }
  }

  /**
   * @brief Update this accumulator with the specified observation.
   *
   * @param y The observed vector.
   * @pre y.size() == mean().size()
   * Eigen will `assert()` in debug builds if the sizes mismatch; behavior is
   * undefined in release builds.
   */
  template <typename Derived>
  inline void observe(const Eigen::MatrixBase<Derived>& y) {
    const VecS delta = y - mean_;
    weight_ = discount_factor_ * weight_ + 1;
    mean_ += delta / weight_;
    sum_sq_dev_ = discount_factor_ * sum_sq_dev_ + delta.cwiseProduct(y - mean_);
  }

  /**
   * @brief Return the estimate of the mean.
   *
   * @return The mean estimate.
   */
  inline const VecS& mean() const noexcept {
    return mean_;
  }

  /**
   * @brief Return the estimate of the variance.
   *
   * @return The variance estimate.
   */
  inline VecS variance() const {
    if (weight_ > 0) {
      return sum_sq_dev_ / weight_;
    }
    return VecS::Zero(mean_.size());
  }

  /**
   * @brief Set the discount factor for previous observations to the specified
   * value.
   *
   * **WARNING**:  There are no checks on this method that `discount_factor
   * in [0, 1]`.
   * 
   * @param discount_factor The past discount factor (between 0 and 1,
   * inclusive).  
   */
  inline void set_discount_factor(S discount_factor) noexcept {
    discount_factor_ = discount_factor;
  }

 private:
  /** The discount factor applied to all previous observations. */
  S discount_factor_;

  /** The combined weight in sample size of all previous observations. */
  S weight_;

  /** The current mean estimate */
  VecS mean_;

  /** 
   * The sum of weighted squared deviation (i.e., current variance estimate
   * scaled by weight).
   */
  VecS sum_sq_dev_;
};

}  // namespace nuts
