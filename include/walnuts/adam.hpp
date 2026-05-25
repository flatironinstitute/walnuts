#pragma once

#include <cmath>

namespace walnuts::detail {

/**
 * The Adam stochastic gradient optimizer specialized for step-size
 * adaptation with a decreasing learning rate schedule.
 *
 * The specialization for step size builds in quadratic error (i.e.,
 * log normal), following Nuts. That is, for observed accept rate
 * `accept_observed` and target accept rate `accept_target`, the error
 * is `-0.5 * (accept_observed - accept_target)^2` and its gradient is
 * `accept_target - accept_observed`.
 *
 * This implementation includes a learning rate schedule that divides
 * the specified learning rate by `pow(t, learn_rate_decay)` in
 * iteration `t` (indexed from 1).  The standard version of Adam uses
 * `learn_decay_rate = 0`, so that the learning rate stays fixed and
 * estimates continue to bounce around with new observations. With
 * stepsize decay, Adam converges as long as `0 < learn_rate_decay <=
 * 1`; Nuts used `learn_rate_decay = 0.75` for dual averaging and
 * `learn_rate_decay=0.5` is a reasonable default for Adam.
 *
 * @see Kingma and Ba (2014; @cite kingma2014adam) for the original
 * Adam algorithm.
 *
 * @see Zou et al. (2019 @cite zou2019sufficient) for the proof of
 * convergence with step-size decay.
 */
class Adam {
 public:
  /**
   * Construct an Adam optimizer from tuning parameters and initialization.
   *
   * @param[in] step_size_init The initial step size.
   * @param[in] accept_rate_target The target acceptance rate.
   * @param[in] learning_rate The learning rate.
   * @param[in] gradient_decay The gradient decay rate.
   * @param[in] sq_gradient_decay The squared gradient decay rate.
   * @param[in] stabilization The estimation stabilization parameter.
   * @param[in] learn_rate_decay The decay exponent for iterations.
   */
  Adam(double step_size_init, double accept_rate_target, double learning_rate,
       double gradient_decay, double sq_gradient_decay, double stabilization,
       double learn_rate_decay)
      : theta_(std::log(step_size_init)),
        m_(0),
        v_(0),
        t_(0),
        gradient_decay_pow_(1),
        sq_gradient_decay_pow_(1),
        target_accept_rate_(accept_rate_target),
        learn_rate_(learning_rate),
        gradient_decay_(gradient_decay),
        sq_gradient_decay_(sq_gradient_decay),
        stabilization_(stabilization),
        learn_rate_decay_(learn_rate_decay) {}

  /**
   * Observe an acceptance probability in (0, 1).
   *
   * @param[in] alpha The acceptance probability.
   * @pre alpha > 0 && alpha < 1
   */
  void operator()(double alpha) noexcept {
    ++t_;
    gradient_decay_pow_ *= gradient_decay_;
    sq_gradient_decay_pow_ *= sq_gradient_decay_;

    double grad = target_accept_rate_ - alpha;

    m_ = gradient_decay_ * m_ + (1 - gradient_decay_) * grad;
    v_ = sq_gradient_decay_ * v_ + (1 - sq_gradient_decay_) * grad * grad;

    double m_hat = m_ / (1 - gradient_decay_pow_);
    double v_hat = v_ / (1 - sq_gradient_decay_pow_);

    double decayed_learn_rate = learn_rate_ / std::pow(t_, learn_rate_decay_);
    double denom = std::sqrt(v_hat) + stabilization_;
    theta_ -= decayed_learn_rate * m_hat / denom;
  }

  /**
   * Return the step size estimate.
   *
   * @return The step size.
   */
  double step_size() const noexcept { return std::exp(theta_); }

 private:
  double theta_;
  double m_;
  double v_;
  double t_;
  double gradient_decay_pow_;
  double sq_gradient_decay_pow_;

  const double target_accept_rate_;
  const double learn_rate_;         // Adam: alpha
  const double gradient_decay_;     // Adam: beta2
  const double sq_gradient_decay_;  // Adam: beta2
  const double stabilization_;      // Adam: epsilon
  const double learn_rate_decay_;
};

}  // namespace walnuts::detail
