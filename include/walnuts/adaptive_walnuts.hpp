#ifndef NUTS_ADAPTIVE_WALNUTS_HPP
#define NUTS_ADAPTIVE_WALNUTS_HPP

#include "dual_average.hpp"
#include "util.hpp"
#include "welford.hpp"

namespace nuts {

template <typename S, class F>
Vec<S> grad(const F& logp_grad_fun, const Vec<S>& theta) {
  Vec<S> g;
  S logp;
  logp_grad_fun(theta, logp, g);
  return std::move(g);
}


template <typename S>
struct MassAdaptConfig {
  MassAdaptConfig(const Vec<S>& mass_init, Integer init_count,
                  S iteration_offset):
    mass_init_(mass_init), init_count_(init_count),
    iteration_offset_(iteration_offset)
  {}

  const Vec<S> mass_init_;
  const S init_count_;
  const S iteration_offset_;
  const S discount_factor_;
};

template <typename S>
struct StepAdaptConfig {
  StepAdaptConfig(S step_size_init,
                  S accept_rate_target,
                  S iteration_offset,
                  S learning_rate,
                  S decay_rate):
      step_size_init_(step_size_init), accept_rate_target_(accept_rate_target),
      iteration_offset_(iteration_offset), learning_rate_(learning_rate),
      decay_rate_(decay_rate)
  {}

  const S step_size_init_;
  const S accept_rate_target_;
  const S iteration_offset_;
  const S learning_rate_;
  const S decay_rate_;
};


template <typename S>
struct WalnutsConfig {
  WalnutsConfig(S log_max_error,
                Integer max_nuts_depth,
                Integer max_step_depth):
      log_max_error_(log_max_error), max_nuts_depth_(max_nuts_depth),
      max_step_depth_(max_step_depth)
  {}

  const S log_max_error_;
  const Integer max_nuts_depth_;
  const Integer max_step_depth_;
};


template <typename S>
class StepAdaptHandler {
 public:
  StepAdaptHandler(S step_size_init, S target_accept_rate, S iteration_offset,
                   S learning_rate, S decay_rate):
      dual_average_(step_size_init, target_accept_rate, iteration_offset,
                    learning_rate, decay_rate)
  {}

  void operator()(S observed_accept_prob) {
    dual_average_.update(observed_accept_prob);
  }

  S step_size() {
    return dual_average_.epsilon();
  }

 private:
  DualAverage<S> dual_average_;
};


template <class F, typename S, class RNG>
class AdaptiveWalnuts {
 public:
  AdaptiveWalnuts(RNG& rng,
                  F& logp_grad,
                  const Vec<S>& theta_init,
                  MassAdaptConfig<S>&& mass_cfg,
                  StepAdaptConfig<S>&& step_cfg,
                  WalnutsConfig<S>&& walnuts_cfg):
      mass_cfg_(mass_cfg),
      step_cfg_(step_cfg),
      walnuts_cfg_(walnuts_cfg),
      rand_(rng),
      logp_grad_(logp_grad),
      theta_(theta_init),
      iteration_(0),
      step_adapt_handler_(step_cfg.step_size_init_, step_cfg.accept_rate_target_,
                          step_cfg.iteration_offset_, step_cfg.learning_rate_,
                          step_cfg.decay_rate_) {
    S discount_factor = 1;
    S iter_offset = mass_cfg.iteration_offset_;
    Vec<S> mean_init = Vec<S>::Zeros(theta_init.size());
    Vec<S> grad_init = grad(logp_grad, theta_init).array().abs().vector();
    mass_adapt_ = {discount_factor, iter_offset,
                   std::move(mean_init), std::move(grad_init)};
  }

  const Vec<S> operator()() {
    auto mass = mass_adapt_.variance();
    auto inv_mass = 1 / mass;
    auto chol_mass = mass.array().sqrt().matrix();
    theta_ = transition_w(rand_, logp_grad_, inv_mass, chol_mass,
        		  step_adapt_handler_.step_size(),
                          walnuts_cfg_.max_nuts_depth_,
                          std::move(theta_), walnuts_cfg_.log_max_error_,
                          step_adapt_handler_);
    mass_adapt_.set_alpha(1 - 1 / (mass_cfg_.iteration_offset_ + iteration_));
    mass_adapt_.update(theta_);
    ++iteration_;
    return theta_;
  }


  WalnutsSampler<F, S, RNG> sampler() {
    return WalnutsSampler<F, S, RNG>(rand_, logp_grad_,
                                     mass_adapt_.inverse_mass_matrix(),
                                     step_adapt_handler_.step_size(),
                                     walnuts_cfg_.max_nuts_depth_,
                                     walnuts_cfg_.log_max_error_,
                                     theta_);
  }

 private:
  const MassAdaptConfig<S> mass_cfg_;
  const StepAdaptConfig<S> step_cfg_;
  const WalnutsConfig<S> walnuts_cfg_;

  Random<S, RNG> rand_;
  F& logp_grad_;

  Vec<S> theta_;
  Integer iteration_;

  StepAdaptHandler<S> step_adapt_handler_;
  DiscountedOnlineMoments<S> mass_adapt_;
};

} // namespace nuts

#endif // NUTS_ADAPTIVE_WALNUTS_HPP
