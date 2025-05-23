#include <cmath>

// REFERENCE: Hoffman, M.D. and Gelman, A., 2014. The No-U-Turn
// sampler: adaptively setting path lengths in Hamiltonian Monte
// Carlo. J. Mach. Learn. Res., 15(1), pp.1593-1623.

namespace nuts {

using Integer = std::int32_t;

/**
 * Configuration for dual averaging algorithm.
 *
 * @param S Type of scalars.
 */
template <typename S>
class DualAvgConfig {
 public:
  /**
   * Construct a dual averaging configuration with the specified values.
   *
   * @param delta Target value (> 0).
   * @param t0 Iteration offset (> 0, default 10).
   * @param gamma Divisor for updates (> 0, default 0.05).
   * @param kappa Power of stepsize for update decay (> 0, default 0.75).
   */
  DualAvgConfig(S delta, S t0 = 10, S gamma = 0.05, S kappa = 0.75):
      delta_(delta),
      t0_(t0),
      gamma_(gamma),
      neg_kappa_(-kappa)
  {}

  S delta_;
  S t0_;
  S gamma_;
  S neg_kappa_;
};

template <typename S>
class DualAvgState {
 public:
  /**
   * Construct a state for dual averaging with the specified log initial value.
   *
   * @param log_epsilon_init Natural log of initial value (finite).
   */
  DualAvgState(S log_epsilon_init):
      log_epsilon_(log_epsilon_init),
      log_epsilon_bar_(0),
      h_bar_(0),
      mu_(std::log(10) + log_epsilon),
      m_(1)
  { }

  S log_epsilon_;
  S log_epsilon_bar_;
  S h_bar_;
  S mu_;
  S m_;
};

/**
 * Update the dual averaging state in the specified iteration
 * given the observed value, and configuration.
 *
 * @param m Iteration number (>= 1).
 * @param alpha Observed value (> 0).
 * @param state Dual averaging state to update.
 * @param cfg Fixed dual averaging configuration.
 */
template <typename S>
void dual_avg_update(S alpha,
                     DualAvgState<S>& state,
                     const DualAvgConfig<S>& cfg) {
  S prop = 1 / (state.m_ + cfg.t0_);
  state.h_bar_ = (1 - prop) * state.h_bar_ + prop * (cfig.delta_ - alpha);
  S last_log_epsilon = state.log_epsilon_;
  state.log_epsilon_ = state.mu_ - std::sqrt(state.m_) / cfg.gamma_ * state.h_bar_;
  S prop2 = std::pow(state.m_, cfg.neg_kappa_);
  state.log_epsilon_bar_ = prop2 * state.log_epsilon_
      + (1 - prop2) * last_log_epsilon;
  ++state.m_;
}


template <typename S>
class DualAvg {
  DualAvgState state_;
  const DualAvgConfig cfg_;

  DualAvg(S epsilon_init, S delta, S t0 = 10, S gamma = 0.05, S kappa = 0.75):
      state_(std::log(epsilon_init)),
      cfg_(delta, t0, gamma, kappa) { }

  void update(S alpha) {
    dual_avg_update(alpha, state_, cfg_);
  }

  S epsilon() const, noexcept {
    return std::exp(state_.log_epsilon_);
  }
}


}  // namespace nuts
