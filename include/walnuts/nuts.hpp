#ifndef NUTS_NUTS_HPP
#define NUTS_NUTS_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <tuple>
#include <utility>

#include <Eigen/Dense>

#include "util.hpp"

namespace nuts {

template <typename S>
class Span {
 public:
  Span(Vec<S> &&theta, Vec<S> &&rho, Vec<S> &&grad_theta, S logp)
      : theta_bk_(theta),
        rho_bk_(rho),
        grad_theta_bk_(grad_theta),
        theta_fw_(theta),
        rho_fw_(std::move(rho)),
        grad_theta_fw_(std::move(grad_theta)),
        theta_select_(std::move(theta)),
        logp_(logp) {}

  Span(Span<S> &&span1, Span<S> &&span2, Vec<S> &&theta_select, S logp)
      : theta_bk_(std::move(span1.theta_bk_)),
        rho_bk_(std::move(span1.rho_bk_)),
        grad_theta_bk_(std::move(span1.grad_theta_bk_)),
        theta_fw_(std::move(span2.theta_fw_)),
        rho_fw_(std::move(span2.rho_fw_)),
        grad_theta_fw_(std::move(span2.grad_theta_fw_)),
        theta_select_(std::move(theta_select)),
        logp_(logp) {}

  Vec<S> theta_bk_;
  Vec<S> rho_bk_;
  Vec<S> grad_theta_bk_;
  Vec<S> theta_fw_;
  Vec<S> rho_fw_;
  Vec<S> grad_theta_fw_;
  Vec<S> theta_select_;
  S logp_;
};

template <typename S, typename F>
void leapfrog(const F &logp_grad_fun, const Vec<S> &inv_mass, S step,
              const Vec<S> &theta, const Vec<S> &rho, const Vec<S> &grad,
              Vec<S> &theta_next, Vec<S> &rho_next, Vec<S> &grad_next,
              S &logp_next) {
  S half_step = 0.5 * step;
  rho_next.noalias() = rho + half_step * grad;
  theta_next.noalias() =
      theta + step * (inv_mass.array() * rho_next.array()).matrix();
  logp_grad_fun(theta_next, logp_next, grad_next);
  rho_next.noalias() += half_step * grad_next;
  logp_next += logp_momentum(rho_next, inv_mass);
}

template <Update U, Direction D, typename S, class Generator>
Span<S> combine(Random<S, Generator> &rng, Span<S> &&span_old,
                Span<S> &&span_new) {
  using std::log;
  S logp_total = log_sum_exp(span_old.logp_, span_new.logp_);
  S log_denominator;
  if constexpr (U == Update::Metropolis) {
    log_denominator = span_old.logp_;
  } else {  // Update::Barker
    log_denominator = logp_total;
  }
  S update_logprob = span_new.logp_ - log_denominator;
  bool update = log(rng.uniform_real_01()) < update_logprob;
  auto &selected = update ? span_new.theta_select_ : span_old.theta_select_;

  auto &&[span_bk, span_fw] = order_forward_backward<D>(span_old, span_new);
  return Span<S>(std::move(span_bk), std::move(span_fw), std::move(selected),
                 logp_total);
}

template <Direction D, typename S, class F>
Span<S> build_leaf(const F &logp_grad_fun, const Span<S> &span,
                   const Vec<S> &inv_mass, S step) {
  Vec<S> theta_next;
  Vec<S> rho_next;
  Vec<S> grad_theta_next;
  S logp_theta_next;
  if constexpr (D == Direction::Forward) {
    leapfrog(logp_grad_fun, inv_mass, step, span.theta_fw_, span.rho_fw_,
             span.grad_theta_fw_, theta_next, rho_next, grad_theta_next,
             logp_theta_next);
  } else {  // Direction::Backward
    leapfrog(logp_grad_fun, inv_mass, -step, span.theta_bk_, span.rho_bk_,
             span.grad_theta_bk_, theta_next, rho_next, grad_theta_next,
             logp_theta_next);
  }
  return Span<S>(std::move(theta_next), std::move(rho_next),
                 std::move(grad_theta_next), logp_theta_next);
}

template <Direction D, typename S, class F, class Generator>
std::optional<Span<S>> build_span(Random<S, Generator> &rng,
                                  const F &logp_grad_fun,
                                  const Vec<S> &inv_mass, S step, Integer depth,
                                  const Span<S> &last_span) {
  if (depth == 0) {
    return build_leaf<D>(logp_grad_fun, last_span, inv_mass, step);
  }
  auto maybe_subspan1 =
      build_span<D>(rng, logp_grad_fun, inv_mass, step, depth - 1, last_span);
  if (!maybe_subspan1) {
    return std::nullopt;
  }
  auto maybe_subspan2 = build_span<D>(rng, logp_grad_fun, inv_mass, step,
                                      depth - 1, *maybe_subspan1);
  if (!maybe_subspan2) {
    return std::nullopt;
  }
  if (uturn<D>(*maybe_subspan1, *maybe_subspan2, inv_mass)) {
    return std::nullopt;
  }
  return combine<Update::Barker, D>(rng, std::move(*maybe_subspan1),
                                    std::move(*maybe_subspan2));
}

template <typename S, class F, class Generator>
Vec<S> transition(Random<S, Generator> &rng, const F &logp_grad_fun,
                  const Vec<S> &inv_mass, const Vec<S> &chol_mass, S step,
                  Integer max_depth, Vec<S> &&theta) {
  Vec<S> rho = rng.standard_normal(theta.size()).cwiseProduct(chol_mass);
  Vec<S> grad(theta.size());
  S logp;
  logp_grad_fun(theta, logp, grad);
  logp += logp_momentum(rho, inv_mass);
  Span<S> span_accum(std::move(theta), std::move(rho), std::move(grad), logp);
  for (Integer depth = 0; depth < max_depth; ++depth) {
    bool go_forward = rng.uniform_binary();
    if (go_forward) {
      constexpr Direction D = Direction::Forward;
      auto maybe_next_span =
          build_span<D>(rng, logp_grad_fun, inv_mass, step, depth, span_accum);
      if (!maybe_next_span) {
        break;
      }
      bool combined_uturn = uturn<D>(span_accum, *maybe_next_span, inv_mass);
      span_accum = combine<Update::Metropolis, D>(rng, std::move(span_accum),
                                                  std::move(*maybe_next_span));
      if (combined_uturn) {
        break;
      }
    } else {
      constexpr Direction D = Direction::Backward;
      auto maybe_next_span =
          build_span<D>(rng, logp_grad_fun, inv_mass, step, depth, span_accum);
      if (!maybe_next_span) {
        break;
      }
      bool combined_uturn = uturn<D>(span_accum, *maybe_next_span, inv_mass);
      span_accum = combine<Update::Metropolis, D>(rng, std::move(span_accum),
                                                  std::move(*maybe_next_span));
      if (combined_uturn) {
        break;
      }
    }
  }
  return std::move(span_accum.theta_select_);
}

/**
 * @brief The NUTS Markov chain Monte Carlo (MCMC) sampler.
 *
 * The sampler is constructed with a base random number generator, a log density
 * and gradient function, an initialization, and several tuning parameters.
 * It provides a no-argument functor for generating the next state in the 
 * Markov chain.
 * 
 * The target log density and gradient function must implement the signature
 * 
 * ```cpp
 * static void normal_logp_grad(const Eigen::Matrix<S, -1, 1>& x,
 *                              S& logp,
 *                              Eigen::Matrix<S, -1, 1>& grad);
 * ```
 * 
 * where `S` is the scalar type parameter of the sampler (the log
 * density function need not be templated itself.  The argument `x`
 * is the position argument, and `logp` is set to the log density of
 * `x`, and `grad` set to the gradient of the log density at `x`.
 * 
 * @tparam F The type of the log density and gradient function.
 * @tparam S The type of scalars.
 * @tparam RNG The type of the base random number generator.
 */  
template <class F, typename S, class RNG>
class Nuts {
 public:
  Nuts(Random<S, RNG>& rand,
       F& logp_grad,
       const Vec<S>& theta_init,
       const Vec<S>& inv_mass,
       S step_size,
       Integer max_nuts_depth):
    rand_(rand), logp_grad_(logp_grad), theta_(theta_init), inv_mass_(inv_mass),
    cholesky_mass_(inv_mass.array().sqrt().inverse().matrix()),
    step_size_(step_size), max_nuts_depth_(max_nuts_depth)
  {}
    
  /**
   * @brief Return the next draw from the sampler.
   *
   * @return The next draw.
   */
  Vec<S> operator()() {
    theta_ = transition(rand_, logp_grad_, inv_mass_, cholesky_mass_,
			step_size_, max_nuts_depth_, std::move(theta_));
    return theta_;
  }

 private:
  /** The underlying randomizer. */
  Random<S, RNG> rand_;

  /** The target log density/gradient function. */
  F& logp_grad_;

  /** The current state. */
  Vec<S> theta_;

  /** The diagonal of the diagonal inverse mass matrix. */
  const Vec<S> inv_mass_;

  /** The diagonal of the diagonal Cholesky factor of the mass matrix. */
  const Vec<S> cholesky_mass_;

  /** The initial step size. */
  const S step_size_;

  /** The maximum number of doublings in NUTS trajectories. */
  const Integer max_nuts_depth_;
};  

}  // namespace nuts

#endif // NUTS_NUTS_HPP
