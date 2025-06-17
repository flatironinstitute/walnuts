#ifndef NUTS_DUAL_AVERAGE_HPP
#define NUTS_DUAL_AVERAGE_HPP

#include <cmath>

// REFERENCE: Hoffman, M.D. and Gelman, A., 2014. The No-U-Turn
// sampler: adaptively setting path lengths in Hamiltonian Monte
// Carlo. J. Mach. Learn. Res., 15(1), pp.1593-1623.

namespace nuts {

/**
 * Class holding configuration and state for a dual averaging
 * estimator.
 *
 * The algorithm is defined in the following reference.
 *
 * \see Hoffman, M.D. and Gelman, A. 2014. The No-U-Turn
 * sampler: adaptively setting path lengths in Hamiltonian Monte
 * Carlo. J. Mach. Learn. Res., 15(1), pp.1593-1623.
 *
 * @tparam S Type of scalars.
 */
template <typename S>
class DualAverage {
public:
  /**
   * Construct a dual averaging estimator with the specified initial
   * value, target value, and tuning parameters.
   *
   * Larger values of `t0` delay fast adaptation and damp early
   * instability.  Smaller values of `gamma` lead to slower, but more
   * stable adaptation.  Higher values of `kappa` slow the freezing of
   * updates and provide more smoothing.
   *
   * The algorithm converges at a logarithmic rate.  It should be
   * reasonably accurate at achieving a target acceptance rate, but
   * the step size estimates themselves tend to be higher variance.
   *
   * @param epsilon_init Initial value (> 0).
   * @param delta Target acceptance rate (> 0).
   * @param t0 Iteration offset (> 0, default 10).
   * @param gamma Learning rate (> 0, default 0.05).
   * @param kappa Decay for averaging (> 0, default 0.75).
   */
  DualAverage(S epsilon_init, S delta, S t0 = 10, S gamma = 0.05,
              S kappa = 0.75):
      // std::cout << "epsilon_init=" << epsilon_init << std::endl;
      log_epsilon_(std::log(epsilon_init)),
      log_epsilon_bar_(0),
      h_bar_(0),
      mu_(std::log(10) + std::log(epsilon_init)),
      m_(1),
      delta_(delta),
      t0_(t0),
      gamma_(gamma),
      neg_kappa_(-kappa) {
    std::cout << "epsilon_init = " << epsilon_init << std::endl;
  }

  /**
   * Update the state for the observed value.
   *
   * @param alpha Observed value (> 0).
   */
  void update(S alpha) noexcept {
    if (std::isnan(alpha)) {  // TODO: figure out why this is sometimes nan
      return;
    }
    S prop = 1 / (m_ + t0_);
    h_bar_ = (1 - prop) * h_bar_ + prop * (delta_ - alpha);
    S last_log_epsilon = log_epsilon_;
    log_epsilon_ = mu_ - std::sqrt(m_) / gamma_ * h_bar_;
    S prop2 = std::pow(m_, neg_kappa_);
    log_epsilon_bar_ = prop2 * log_epsilon_
        + (1 - prop2) * last_log_epsilon;
    ++m_;
  }

  /**
   * Return the current estimate of epsilon.
   *
   * @return Estimate of epsilon.
   */
  S epsilon() const noexcept {
    return std::exp(log_epsilon_bar_);
  }

private:
  S log_epsilon_;
  S log_epsilon_bar_;
  S h_bar_;
  S mu_;
  S m_;
  const S delta_;
  const S t0_;
  const S gamma_;
  const S neg_kappa_;
};

}  // namespace nuts

#endif // NUTS_DUAL_AVERAGE_HPP
