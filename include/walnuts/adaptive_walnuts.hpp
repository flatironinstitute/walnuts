#include <nuts/util.hpp>

namespace nuts {

template <typename S>
struct MassAdaptConfig {
  MassAdaptConfig(const Vec<S>& mass_init,
                  S iteration_offset):
      mass_init_(mass_init), iteration_offset_(iteration_offset)
  {}

  const Vec<S> mass_init_;
  const S iteration_offset_;
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
  const S iteration_offet_;
  const S learning_rate_;
  const S decay_rate_;
};

template <typename S>
struct WalnutsConfig {
  WalnutsConifg(S log_min_accept,
                Intger max_nuts_depth,
                Integer max_step_depth):
      log_min_accept_(log_min_accept), max_nuts_depth_(max_nuts_depth),
      max_step_depth_(max_step_depth)
  {}

  const S log_min_accept_;
  const Integer max_nuts_depth_;
  const Integer max_step_depth_;
};

template <class F, typename S, RNG = std::mt19937>
class WalnutsSampler {
 public:
  WalnutsSampler(RNG& rng, F& logp_grad,
                 const Vec<S>& inverse_mass_matrix,
                 S macro_step_size,
                 const WalnutsConfig& walnuts_cfg
                 const Vec<S>& theta):
      rng_(rng), logp_grad_(logp_grad),
      inverse_mass_matrix_(inverse_mass_matrix),
      step_size_(macro_step_size), walnuts_cfg_(walnuts_cfg)
  { }

  Vec<S> operator()();

 private:
  RNG rng_;
  F& logp_grad_
  Vec<S> theta_;
  const Vec<S> inv_mass_matrix_;
  const S step_size_;
  const WalnutsConfig walnuts_cfg;
};


template <class F, typename S, class RNG = std::mt19937>
class AdaptiveWalnuts {
 public:
  AdaptiveWalnuts(RNG& rng,
                  F& logp_grad,
                  const Vec<S>& theta_init,
                  const MassAdaptConfig& mass_cfg,
                  const StepAdaptConfig& step_cfg,
                  const WalnutsConfig& walnuts_cfg):
      rng_(rng), logp_grad_(logp_grad), theta_(theta_init),
      mass_cfg_(mass_cfg), step_cfg_(step_cfg), walnuts_cfg_(walnuts_cfg)
  { }

  Vec<S> operator()();

  WalnutsSampler sampler() {
    return WalnutsSampler(rng_, logp_grad, mass_adapt_.inverse_mass_matrix(),
                          step_adapt.step_size(), walnuts_cfg_, theta_);
  }

 private:
  Vec<S> theta_;
  F& logp_grad_
  RNG rng_;

  MassAdapt mass_adapt_;
  StepAdapt step_adapt_;

  const F logp_grad_;
  const WalnutsConfig walnuts_cfg_;
};
