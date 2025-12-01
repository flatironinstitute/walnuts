#pragma once

#include <cmath>
#include <stdexcept>

namespace nuts {

template <typename S>
class Adam {
 public:
  Adam(S step_size_init, S target_accept_rate, S learn_rate = 0.2,
       S beta1 = 0.3, S beta2 = 0.99, S eps = 1e-8)
      : theta_(std::log(step_size_init)),
        m_(0),
        v_(0),
        t_(0),
        beta1_pow_(1),
        beta2_pow_(1),
        target_accept_rate_(target_accept_rate),
        learn_rate_(learn_rate),
        beta1_(beta1),
        beta2_(beta2),
        eps_(eps) {
    if (!(step_size_init > 0 && std::isfinite(step_size_init))) {
      throw std::invalid_argument(
          "Initial step_size must be positive and finite.");
    }
    if (!(target_accept_rate > 0 && std::isfinite(target_accept_rate))) {
      throw std::invalid_argument(
          "Target acceptance rate must be positive and finite.");
    }
    if (!(learn_rate > 0 && std::isfinite(learn_rate))) {
      throw std::invalid_argument("Learning rate must be positive and finite.");
    }
    if (!(beta1 > 0 && beta1 < 1)) {
      throw std::invalid_argument("beta1 must be in (0, 1) and finite.");
    }
    if (!(beta2 > 0 && beta2 < 1)) {
      throw std::invalid_argument("beta2 must be in (0, 1) and finite.");
    }
    if (!(eps > 0 && std::isfinite(eps))) {
      throw std::invalid_argument("eps must be positive and finite.");
    }
  }

  inline void observe(S alpha) noexcept {
    ++t_;
    beta1_pow_ *= beta1_;
    beta2_pow_ *= beta2_;

    const S grad = target_accept_rate_ - alpha;

    m_ = beta1_ * m_ + (1 - beta1_) * grad;
    v_ = beta2_ * v_ + (1 - beta2_) * grad * grad;

    const S m_hat = m_ / (1 - beta1_pow_);
    S v_hat = v_ / (1 - beta2_pow_);

    if (v_hat < 0) {
      v_hat = 0;
    }

    const S denom = std::sqrt(v_hat) + eps_;
    const S effective_lr = learn_rate_ / std::sqrt(static_cast<S>(t_));
    theta_ -= effective_lr * m_hat / denom;
  }

  inline S step_size() const noexcept { return std::exp(theta_); }

 private:
  S theta_;
  S m_;
  S v_;
  std::size_t t_;
  S beta1_pow_;
  S beta2_pow_;

  const S target_accept_rate_;
  const S learn_rate_;
  const S beta1_;
  const S beta2_;
  const S eps_;
};

}  // namespace nuts
