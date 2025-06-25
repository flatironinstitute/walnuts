#pragma once

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

/**
 * @brief A span encodes the necessary information about a Hamiltonian
 * trajectory for Hamiltonian Monte Carlo with computation caching.
 * 
 * @tparam S The type of scalars.
 */
template <typename S>
class Span {
 public:
  /** @brief Construct a span of a single state.
   *
   * @param[in] theta The position.
   * @param[in] rho The momentum.
   * @param[in] grad_theta The gradient of the target log density at the
   * position. 
   * @param[in] logp The joint log density of the position and momentum.
   */
  Span(Vec<S> &&theta, Vec<S> &&rho, Vec<S> &&grad_theta, S logp)
      : theta_bk_(theta),
        rho_bk_(rho),
        grad_theta_bk_(grad_theta),
        theta_fw_(theta),
        rho_fw_(std::move(rho)),
        grad_theta_fw_(std::move(grad_theta)),
        theta_select_(std::move(theta)),
        logp_(logp) {}
  /**
   * @brief Construct a span by concatening the specified spans with
   * the specified selected state.
   *
   * @param[in] span1 The first span in temporal order.
   * @param[in] span2 The second span in temporal order.
   * @param[in] theta_select The selected position.
   * @param[in] logp The log of the sum of the denisites on the trajectory.
   */
  Span(Span<S> &&span1, Span<S> &&span2, Vec<S> &&theta_select, S logp)
      : theta_bk_(std::move(span1.theta_bk_)),
        rho_bk_(std::move(span1.rho_bk_)),
        grad_theta_bk_(std::move(span1.grad_theta_bk_)),
        theta_fw_(std::move(span2.theta_fw_)),
        rho_fw_(std::move(span2.rho_fw_)),
        grad_theta_fw_(std::move(span2.grad_theta_fw_)),
        theta_select_(std::move(theta_select)),
        logp_(logp) {}

  /** The earliest position. */
  Vec<S> theta_bk_;

  /** The earliest momentum. */
  Vec<S> rho_bk_;

  /** The gradient of the target log density at the earliest position. */
  Vec<S> grad_theta_bk_;

  /** The latest state. */
  Vec<S> theta_fw_;

  /** The latest momentum. */
  Vec<S> rho_fw_;

  /** The gradient of the target log density at the latest position. */
  Vec<S> grad_theta_fw_;

  /** The selected state. */
  Vec<S> theta_select_;

  /** The log of the sum of the densities on the trajectory. */
  S logp_;
};

/**
 * @brief Perfrom one step of the leapfrog algorithm for simulating
 * Hamiltonians.
 *
 * @tparam S The type of scalars.
 * @tparam F The type of the target log density/gradient function.
 * @param[in,out] logp_grad_fun The target log density/gradient function.
 * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix
 * (finite positive components).
 * @param[in] theta Starting position.
 * @param[in] rho Starting momentum.
 * @param[in] grad Gradient of target log densitity at the starting position.
 * @param[out] theta_next The ending position.
 * @param[out] rho_next The ending momentum.
 * @param[out] grad_next The gradient of the log density at the ending position.
 * @param[out] logp_next The joint log density of the ending poisiton and
 * momentum.
 */  
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

/**
 * @brief Return the concatenation of the specified spans in the
 * specified temporal direction.
 *
 * @tparam U The acceptance rule to update the selected state.
 * @tparam D The direction of temporal ordering.
 * @tparam S The type of scalars.
 * @tparam RNG The base random number generator.
 * @param[in,out] rand The compound random number generator.
 * @param[in] span_old The first span in the ordering.
 * @param[in] span_new The new span in the ordering.
 * @return The spans combined in the specified temporal ordering.
 */  
template <Update U, Direction D, typename S, class RNG>
Span<S> combine(Random<S, RNG> &rand, Span<S> &&span_old,
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
  bool update = log(rand.uniform_real_01()) < update_logprob;
  auto &selected = update ? span_new.theta_select_ : span_old.theta_select_;

  auto &&[span_bk, span_fw] = order_forward_backward<D>(span_old, span_new);
  return Span<S>(std::move(span_bk), std::move(span_fw), std::move(selected),
                 logp_total);
}

/**
 * @brief Return the span consisting of one state that follows the specified
 * span in the specified temporal direction.
 * 
 * @tparam D The temporal direction in which to extend the last span.
 * @tparam S The type of scalars.
 * @tparam F The type of the target log density/gradient function.
 * @param[in,out] logp_grad_fun The target log density/gradient function.
 * @param[in] span The span to extend.
 * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix
 * (positive finite components).
 * @param[in] step The step size (finite positive floating point).
 * @return The single-state span that follows the specified span.
 */
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

/**
 * @brief If possible, return a span of the specified depth that extends
 * the last span in the specified direction.
 *
 * If there is a sub-U-turn within the newly built span, the return will 
 * be `std::nullopt`.  The randomizer is required to update states.
 *
 * @tparam D The temporal direction in which to extend the last span.
 * @tparam F The type of the target log density/gradient function.
 * @tparam RNG The type of the base random number generator.
 * @param[in,out] rand The compound random number generator.
 * @param[in,out] logp_grad_fun The target log density/gradient function.
 * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix (finite
 * positive components).
 * @param[in] step The step size (finite positive floating point).
 * @param[in] depth The number of trajectory doublings (non-negative integer).
 * @param[in] last_span The span to extend.
 * @return The new span of `std::nullopt` if there was a sub-u-turn.
 */
template <Direction D, typename S, class F, class RNG>
std::optional<Span<S>> build_span(Random<S, RNG> &rand,
                                  const F &logp_grad_fun,
                                  const Vec<S> &inv_mass, S step, Integer depth,
                                  const Span<S> &last_span) {
  if (depth == 0) {
    return build_leaf<D>(logp_grad_fun, last_span, inv_mass, step);
  }
  auto maybe_subspan1 =
      build_span<D>(rand, logp_grad_fun, inv_mass, step, depth - 1, last_span);
  if (!maybe_subspan1) {
    return std::nullopt;
  }
  auto maybe_subspan2 = build_span<D>(rand, logp_grad_fun, inv_mass, step,
                                      depth - 1, *maybe_subspan1);
  if (!maybe_subspan2) {
    return std::nullopt;
  }
  if (uturn<D>(*maybe_subspan1, *maybe_subspan2, inv_mass)) {
    return std::nullopt;
  }
  return combine<Update::Barker, D>(rand, std::move(*maybe_subspan1),
                                    std::move(*maybe_subspan2));
}

/**
 * @brief Return the next state in the Markov chain given a randomizer,
 * target log density, current position, and tuning parameters.
 *
 * @tparam S The type of scalars.
 * @tparam F The type of the log density/gradient function.
 * @tparam RNG The type of the base random number generator.
 * @param[in,out] rand The compound random number generator.
 * @param[in,out] logp_grad_fun The target log density/gradient function.
 * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix.
 * @param[in] chol_mass The diagonal of the diagonal Cholesky factor of the mass
 * matrix.
 * @param[in] step The step size.
 * @param[in] max_depth The maxmium number of doublings of the trajectory.
 * @param[in] theta The current state.
 * @return The next state in the NUTS Markov chain.
 */
template <typename S, class F, class RNG>
Vec<S> transition(Random<S, RNG> &rand, const F &logp_grad_fun,
                  const Vec<S> &inv_mass, const Vec<S> &chol_mass, S step,
                  Integer max_depth, Vec<S> &&theta) {
  Vec<S> rho = rand.standard_normal(theta.size()).cwiseProduct(chol_mass);
  Vec<S> grad(theta.size());
  S logp;
  logp_grad_fun(theta, logp, grad);
  logp += logp_momentum(rho, inv_mass);
  Span<S> span_accum(std::move(theta), std::move(rho), std::move(grad), logp);
  for (Integer depth = 0; depth < max_depth; ++depth) {
    bool go_forward = rand.uniform_binary();
    if (go_forward) {
      constexpr Direction D = Direction::Forward;
      auto maybe_next_span =
          build_span<D>(rand, logp_grad_fun, inv_mass, step, depth, span_accum);
      if (!maybe_next_span) {
        break;
      }
      bool combined_uturn = uturn<D>(span_accum, *maybe_next_span, inv_mass);
      span_accum = combine<Update::Metropolis, D>(rand, std::move(span_accum),
                                                  std::move(*maybe_next_span));
      if (combined_uturn) {
        break;
      }
    } else {
      constexpr Direction D = Direction::Backward;
      auto maybe_next_span =
          build_span<D>(rand, logp_grad_fun, inv_mass, step, depth, span_accum);
      if (!maybe_next_span) {
        break;
      }
      bool combined_uturn = uturn<D>(span_accum, *maybe_next_span, inv_mass);
      span_accum = combine<Update::Metropolis, D>(rand, std::move(span_accum),
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
 * and gradient function, an initial position, and several tuning parameters.
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
  /**
   * @brief Construct a NUTS sampler from the specified randomizer, target
   * log density, initialization, and tuning parameters.
   *
   * @param[in,out] rand The compound random number generator for MCMC.
   * @param[in,out] logp_grad The target log density/gradient function.
   * @param[in] theta_init The initial position.
   * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix (finite
   * positive comonents).
   * @param [in] step_size The step size (finite positive floating point).
   * @param [in] max_nuts_depth The maximum number of trajectory doublings
   * in NUTS (positive integer).
   */
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
