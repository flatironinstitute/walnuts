#pragma once

#include <cmath>
#include <optional>
#include <utility>

#include <Eigen/Dense>

#include "util.hpp"

namespace nuts {

/**
 * @brief A class for holding the minimal information in a Hamiltonian
 * trajectory required for WALNUTS.
 *
 * A span has member variables for the initial and final states' (a)
 * position, (b) momentum, (c) log density of the state, and (d)
 * gradient of target log density.  It also holds a selected state,
 * the gradient of the selected state, and the log of the sum of all
 * joint densities on the trajectory. The gradients could be recomputed,
 * but storing them serves as a local cache.
 *
 * @tparam S Type of scalars.
 */
template <typename S>
class SpanW {
 public:
  /**
   * @brief Construct a span of one state given the specified
   * position, momentum, gradient, and log density.
   *
   * @param[in] theta The position.
   * @param[in] rho The momentum.
   * @param[in] grad_theta The gradient of the log density at `theta`.
   * @param[in] logp The joint log density of the position and momentum.
   */
  static SpanW<S> from_initial_point(Vec<S>&& theta, Vec<S>&& rho,
                                     Vec<S>&& grad_theta, S logp) {
    return {theta,
            rho,
            grad_theta,
            logp,
            theta,
            std::move(rho),
            grad_theta,
            logp,
            std::move(theta),
            std::move(grad_theta),
            logp};
  }

  /**
   * @brief Construct a span by concatenating the two specified spans
   * with the given state selected and total log density.
   *
   * @param[in] span1 The earlier span by temporal ordering.
   * @param[in] span2 The later span by temporal ordering.
   * @param[in] theta_select The selected position.
   * @param[in] grad_select The gradient of the target log density at the
   * selected position.
   * @param[in] logp The log of the sum of the densities on the trajectory.
   */
  static SpanW<S> from_subspans(SpanW<S>&& span1, SpanW<S>&& span2,
                                Vec<S>&& theta_select, Vec<S>&& grad_select,
                                S logp) {
    return {std::move(span1.theta_bk_),
            std::move(span1.rho_bk_),
            std::move(span1.grad_theta_bk_),
            span1.logp_bk_,
            std::move(span2.theta_fw_),
            std::move(span2.rho_fw_),
            std::move(span2.grad_theta_fw_),
            span2.logp_fw_,
            std::move(theta_select),
            std::move(grad_select),
            logp};
  }

  /** The earliest state. */
  Vec<S> theta_bk_;

  /** The earliest momentum. */
  Vec<S> rho_bk_;

  /** The gradient of the target log density at the earliest state . */
  Vec<S> grad_theta_bk_;

  /** The joint log density of the earliest position and momentum. */
  S logp_bk_;

  /** The latest state in the trajectory. */
  Vec<S> theta_fw_;

  /** The latest momentum in the trajectory. */
  Vec<S> rho_fw_;

  /** The gradient of the target log density at the latest position. */
  Vec<S> grad_theta_fw_;

  /** The joint log density of the latest position and momentum. */
  S logp_fw_;

  /** The selected state. */
  Vec<S> theta_select_;

  /** The gradient of the log density at the selected state. */
  Vec<S> grad_select_;

  /** The log of the sum of the joint densities in the trajectory. */
  S logp_;
};

/**
 * @brief Return `true` if running the specified number of leapfrog steps
 * is within the maximum error tolerance.
 *
 * @tparam S The type of scalars.
 * @tparam F The type of the log density/gradient function.
 * @param[in] logp_grad The log density/gradient function.
 * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix.
 * @param[in] step The micro step size.
 * @param[in] num_steps The number of micro steps to take.
 * @param[in] max_error The maximum error in Hamiltonian at macro steps.
 * @param[in] logp_next Initial log density.
 * @param[in,out] theta_next Input initial position, set to final position.
 * @param[in,out] rho_next Input initial momentum, set to final position.
 * @param[in,out] grad_next Input initial gradient, set to final gradient.
 */
template <typename S, typename F>
bool within_tolerance(const F& logp_grad, const Vec<S>& inv_mass, S step,
                      std::size_t num_steps, S max_error, S logp_next,
                      Vec<S>& theta_next, Vec<S>& rho_next, Vec<S>& grad_next) {
  S half_step = 0.5 * step;
  S logp = logp_next;
  for (std::size_t n = 0; n < num_steps; ++n) {
    rho_next.noalias() = rho_next + half_step * grad_next;
    theta_next.noalias() +=
        step * (inv_mass.array() * rho_next.array()).matrix();
    logp_grad(theta_next, logp_next, grad_next);
    rho_next.noalias() += half_step * grad_next;
  }
  logp_next += logp_momentum(rho_next, inv_mass);
  return std::abs(logp_next - logp) <= max_error;
}

/**
 * @brief Return `true` if the number of micro steps provided is the one chosen
 * from the input position, moment, and gradient.
 *
 * @tparam S Type of scalars.
 * @tparam F Type of log density/gradient function.
 * @param[in] logp_grad The log density/gradient function.
 * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix.
 * @param[in] step The micro step size.
 * @param[in] num_steps The number of micro steps proposed forward.
 * @param[in] min_micro_steps The minimum number of micro steps to take.
 * @param[in] max_error The maximum error tolerance in Hessians.
 * @param[in] logp_next The log density of the starting position.
 * @param[in] theta The final position from which to reverse.
 * @param[in] rho The final momentum from which to reverse.
 * @param[in] grad The final gradient from which to reverse.
 * @return `true` if the path ending in the specified state is reversible.
 */
template <typename S, typename F>
bool reversible(const F& logp_grad, const Vec<S>& inv_mass, S step,
                std::size_t num_steps, std::size_t min_micro_steps, S max_error, S logp_next,
                const Vec<S>& theta, const Vec<S>& rho, const Vec<S>& grad) {
  if (num_steps == 1) {
    return true;
  }
  Vec<S> theta_next(grad.size());
  Vec<S> rho_next(rho.size());
  Vec<S> grad_next(grad.size());
  while (num_steps > 2 * min_micro_steps) {
    theta_next = theta;
    rho_next = -rho;
    grad_next = grad;
    num_steps /= 2;
    step *= 2;
    if (within_tolerance(logp_grad, inv_mass, step, num_steps, 
			 max_error, logp_next, theta_next, rho_next, grad_next)) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Take a macro step from the specified state given the log
 * density/gradient, tuning parameters and adaptation handler and
 * return whether it conserves the Hamiltonian and is reversible.
 *
 * @tparam D The time direction of Hamiltonian simulation.
 * @tparam S The type of scalars.
 * @tparam F The type of the log density/gradient function.
 * @tparam A The type of the adaptation handler.
 * @param[in] logp_grad The target log density/gradient function.
 * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix.
 * @param[in] step The initial micro step size.
 * @param[in] max_step_halvings The maximum number of halvings of the step size.
 * @param[in] max_error The maximum difference in Hamiltonians allowed in macro
 * steps. 
 * @param[in] span The span to extend.
 * @param[out] theta_next The position after the macro step.
 * @param[out] rho_next The momentum after the macro step.
 * @param[out] grad_next The gradient of the position after the macro step.
 * @param[out] logp_next The log density of the positon and momentum after the
 * macro step.
 * @param[in,out] adapt_handler The step-size adaptation handler.
 * @return `true` if the Hamiltonian is conserved reversibly.
 */
template <Direction D, typename S, typename F, class A>
bool macro_step(const F& logp_grad, const Vec<S>& inv_mass, S step,
		std::size_t max_step_halvings, std::size_t min_micro_steps,
		S max_error, const SpanW<S>& span, Vec<S>& theta_next,
		Vec<S>& rho_next, Vec<S>& grad_next, S& logp_next,
		A& adapt_handler) {
  using std::fmax, std::fmin;
  constexpr bool is_forward = (D == Direction::Forward);
  const Vec<S>& theta = is_forward ? span.theta_fw_ : span.theta_bk_;
  const Vec<S>& rho = is_forward ? span.rho_fw_ : span.rho_bk_;
  const Vec<S>& grad = is_forward ? span.grad_theta_fw_ : span.grad_theta_bk_;
  S logp = is_forward ? span.logp_fw_ : span.logp_bk_;
  step = is_forward ? step : -step;
  for (std::size_t num_steps = min_micro_steps, halvings = 0; halvings < max_step_halvings;
       ++halvings, num_steps *= 2, step *= 0.5) {
    theta_next = theta;
    rho_next = rho;
    grad_next = grad;
    S half_step = 0.5 * step;
    for (std::size_t n = 0; n < num_steps; ++n) {
      rho_next.noalias() += half_step * grad_next;
      theta_next.noalias() +=
          step * (inv_mass.array() * rho_next.array()).matrix();
      logp_grad(theta_next, logp_next, grad_next);
      rho_next.noalias() += half_step * grad_next;
    }
    logp_next += logp_momentum(rho_next, inv_mass);
    if (num_steps == min_micro_steps) {
      S min_accept = std::exp(-std::fabs(logp - logp_next));
      adapt_handler(min_accept);
    }
    if (std::fabs(logp - logp_next) <= max_error) {
      return reversible(logp_grad, inv_mass, step, num_steps,
			min_micro_steps,
			max_error, logp_next, theta_next, rho_next, grad_next);
    }
  }
  return false;
}

/**
 * @brief Return the specified spans into a new span and select a new position.
 * state.
 *
 * If the direction `D` is `Forward`, then `span_new` is ordered after
 * `span_old` in time; if it is `Backward`, then `span_new` is before
 * `span_old`.
 *
 * The new selected state is determined with either a
 * Metropolis update rule or a Barker update rule based on the
 * template parameter, using the specified random number generator.
 *
 * @tparam U The type of update (`Metropolis` or `Barker`).
 * @tparam D The direction of combination in time (`Forward` or `Backward`).
 * @tparam S The type of scalars.
 * @tparam Rand The type for the source of randomness.
 * @param rng The random number generator used to select a new position.
 * @param span_old The old span.
 * @param span_new The span continuing the old span forward or backward in time.
 * @return The combined span.
 */
template <Update U, Direction D, typename S, class Rand>
SpanW<S> combine(Rand& rng, SpanW<S>&& span_old, SpanW<S>&& span_new) {
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
  auto& selected = update ? span_new.theta_select_ : span_old.theta_select_;
  auto& grad_selected = update ? span_new.grad_select_ : span_old.grad_select_;
  auto&& [span_bk, span_fw] = order_forward_backward<D>(span_old, span_new);
  return SpanW<S>::from_subspans(std::move(span_bk), std::move(span_fw),
                                 std::move(selected), std::move(grad_selected),
                                 logp_total);
}

/**
 * @brief Extend the specified span with a span of a single state.
 *
 * Given the specified span and direction `D`, build a new leaf span consisting
 * of a single state.  If `D` is `Forward`, the leaf extends the specified span
 * forward in time; if `Backward, it extends the span backward in time.
 *
 * The step-size adaptation handler is called with the acceptance of each
 * macro step attempt.
 *
 * The step size is reduced so that the Hamiltonian is conserved
 * within the specified error.  The mass matrix and macro step size
 * are passed on to the leapfrog algorithm.
 *
 * The result is `std::optional` and will be `std::nullopt` only if the
 * specified span could not be extended reversibly within the error threshold.
 *
 * @tparam D The direction in time to extend.
 * @tparam S The type of scalars.
 * @tparam F The type of the log density/gradient function.
 * @tparam A The type of the adaptation handler.
 * @param[in] logp_grad The log density/gradient function.
 * @param[in] span The span to extend.
 * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix.
 * @param[in] step The macro step size.
 * @param[in] max_step_halvings The maximum number of halvings of the step size.
 * @param[in] min_micro_steps The minimum number of micro steps per macro step.
 * @param[in] max_error The maximum error allowed in the Hamiltonian.
 * @param[in,out] adapt_handler The step-size adaptation handler.
 * @return The span resulting from extending the specified span or
 * `std::nullopt` if that could not be done reversibly within threshold.
 */
template <Direction D, typename S, class F, class A>
std::optional<SpanW<S>> build_leaf(const F& logp_grad, const SpanW<S>& span,
                                   const Vec<S>& inv_mass, S step,
				   std::size_t max_step_halvings,
				   std::size_t min_micro_steps,
				   S max_error, A& adapt_handler) {
  Vec<S> theta_next;
  Vec<S> rho_next;
  Vec<S> grad_theta_next;
  S logp_theta_next;
  if (!macro_step<D>(logp_grad, inv_mass, step, min_micro_steps,
		     max_step_halvings, max_error,
		     span, theta_next, rho_next, grad_theta_next,
		     logp_theta_next, adapt_handler)) {
    return std::nullopt;
  }
  return SpanW<S>::from_initial_point(
      std::move(theta_next), std::move(rho_next), std::move(grad_theta_next),
      logp_theta_next);
}

/**
 * @brief Return a span of two to the power of the depth states extending from
 * the specified span, returning `nullopt` if there is a U-turn at any point.
 *
 * @tparam D The direction in time to extend.
 * @tparam S The type of scalars.
 * @tparam F The type of the log density/gradient function.
 * @tparam Rand The type for the source of randomness.
 * @tparam A The type of the step-size adaptation callback function.
 * @param[in,out] rng The random number generator.
 * @param[in] logp_grad The log density/gradient function.
 * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix.
 * @param[in] step The macro step size.
 * @param[in] depth The maximum NUTS depth.
 * @param[in] max_step_halvings The maximum number of halvings of the step size.
 * @param[in] min_micro_steps The minimum number of micro steps per macro step.
 * @param[in] max_error The maximum error allowed at macro steps.
 * @param[in] last_span The span to extend.
 * @param[in,out] adapt_handler The step-size adaptation handler.
 * @return The new span or `std::nullopt` if it could not be constructed.
 */
template <Direction D, typename S, class F, class Rand, class A>
std::optional<SpanW<S>> build_span(Rand& rng, const F& logp_grad,
                                   const Vec<S>& inv_mass, S step,
                                   std::size_t depth, std::size_t max_step_halvings,
				   std::size_t min_micro_steps,
				   S max_error, const SpanW<S>& last_span,
                                   A& adapt_handler) {
  if (depth == 0) {
    return build_leaf<D>(logp_grad, last_span, inv_mass, step, max_step_halvings,
			 min_micro_steps, max_error, adapt_handler);
  }
  auto maybe_subspan1 = build_span<D>(rng, logp_grad, inv_mass, step, depth - 1,
                                      max_step_halvings, min_micro_steps, max_error,
				      last_span, adapt_handler);
  if (!maybe_subspan1) {
    return std::nullopt;
  }
  auto maybe_subspan2 =
    build_span<D>(rng, logp_grad, inv_mass, step, depth - 1, max_step_halvings,
		  min_micro_steps, max_error, *maybe_subspan1, adapt_handler);
  if (!maybe_subspan2) {
    return std::nullopt;
  }
  if (uturn<D>(*maybe_subspan1, *maybe_subspan2, inv_mass)) {
    return std::nullopt;
  }
  return std::make_optional(combine<Update::Barker, D>(
      rng, std::move(*maybe_subspan1), std::move(*maybe_subspan2)));
}

/**
 * @brief Return the next state in the Markov chain given the previous state.
 *
 * @tparam S The type of scalars.
 * @tparam F The type of the log density/gradient function.
 * @tparam Rand The type for the source of randomness.
 * @tparam A The type of the step-size adaptation callback function.
 * @param[in,out] rand The random number generator.
 * @param[in] logp_grad The log density/gradient function.
 * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix.
 * @param[in] chol_mass The diagonal of the diagonal Cholesky factor of the mass
 * matrix.
 * @param[in] step The macro step size.
 * @param[in] max_depth The maximum number of trajectory doublings in NUTS.
 * @param[in] max_step_halvings The maximum number of halvings of the step size.
 * @param[in] min_micro_steps The minimum number of micro steps per macro step.
 * @param[in] max_error The maximum difference in Hamiltonians.
 * @param[in] theta The previous state.
 * @param[out] theta_grad The gradient of the log density at the previous state.
 * @param[in,out] adapt_handler The step-size adaptation handler.
 * @return The next position in the Markov chain.
 */
template <typename S, class F, class Rand, class A>
Vec<S> transition_w(Rand& rand, const F& logp_grad, const Vec<S>& inv_mass,
                    const Vec<S>& chol_mass, S step, std::size_t max_depth,
		    std::size_t max_step_halvings, std::size_t min_micro_steps,
                    S max_error, Vec<S>&& theta, Vec<S>& theta_grad, 
                    A& adapt_handler) {
   std::size_t dims = static_cast<std::size_t>(theta.size());
  Vec<S> rho = rand.standard_normal(dims).cwiseProduct(chol_mass);
  Vec<S> grad(theta.size());
  S logp;
  logp_grad(theta, logp, grad);
  logp += logp_momentum(rho, inv_mass);
  auto span_accum = SpanW<S>::from_initial_point(
      std::move(theta), std::move(rho), std::move(grad), logp);
  for (std::size_t depth = 0; depth < max_depth; ++depth) {
    // helper to turn runtime direction into compile-time template enum
    auto expand_in_direction = [&](auto direction) -> bool {
      constexpr Direction D = direction;
      auto maybe_next_span
      =  build_span<D>(rand, logp_grad, inv_mass, step, depth, max_step_halvings,
		       min_micro_steps, max_error, span_accum, adapt_handler);
      if (!maybe_next_span) {
        return true;
      }
      bool combined_uturn = uturn<D>(span_accum, *maybe_next_span, inv_mass);
      span_accum = combine<Update::Metropolis, D>(rand, std::move(span_accum),
                                                  std::move(*maybe_next_span));
      return combined_uturn;
    };

    bool go_forward = rand.uniform_binary();
    bool made_uturn = go_forward ? expand_in_direction(Forward_t{})
                                 : expand_in_direction(Backward_t{});

    if (made_uturn) {
      break;
    }
  }
  theta_grad = span_accum.grad_select_;
  return std::move(span_accum.theta_select_);
}

/**
 * @brief A functor of one argument that does nothing.
 *
 * The use is as an adaptation handler when there is no adaptation. Because
 * it has no body, it will be inlined away at optimization level `-O2` or
 * above.
 */
class NoOpHandler {
 public:
  /**
   * Do nothing.
   *
   * @tparam T The type of the functor argument.
   */
  template <typename T>
  inline void operator()(const T&) const noexcept {}
};

/**
 * @brief The WALNUTS Markov chain Monte Carlo (MCMC) sampler.
 *
 * The sampler is constructed with a base random number generator, a log density
 * and gradient function, an initialization, and several tuning parameters.
 * It provides a no-argument functor for generating the next element of the
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
class WalnutsSampler {
 public:
  /**
   * @brief Construct a WALNUTS sampler from the specified RNG, target log
   * density/gradient initialization, and tuning parameters.
   *
   * @param[in] rand The randomizer for HMC.
   * @param[in] logp_grad The target log density and gradient function (see the
   * class documentation.
   * @param[in] theta The initial position.
   * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix.
   * @param[in] macro_step_size The initial (largest) step size.
   * @param[in] max_nuts_depth The maximum number of trajectory doublings for NUTS.
   * @param[in] max_step_halvings The maximum number of times the step size is halved.
   * @param[in] min_micro_steps The minimum number of micro steps per macro step.
   * @param[in] max_error The log of the maximum error in joint densities
   * allowed in Hamiltonian trajectories.
   * @throw std::invalid_argument If the max error is not finite and
   * positive.
   * @throw std::invalid_argument If the maximum NUTS depth is zero.
   * @throw std::invalid_argument If the minimum number of micro steps is zero.
   */
  WalnutsSampler(Random<S, RNG>& rand, const F& logp_grad, const Vec<S>& theta,
                 const Vec<S>& inv_mass, S macro_step_size,
                 std::size_t max_nuts_depth, std::size_t max_step_halvings,
		 std::size_t min_micro_steps, S max_error)
      : rand_(rand),
        logp_grad_(logp_grad),
        theta_(theta),
        inv_mass_(inv_mass),
        cholesky_mass_(inv_mass.array().sqrt().inverse().matrix()),
        macro_step_size_(macro_step_size),
        max_nuts_depth_(max_nuts_depth),
	max_step_halvings_(max_step_halvings),
	min_micro_steps_(min_micro_steps),
        max_error_(max_error),
        no_op_adapt_handler_() {
    if (!(max_error > 0) || std::isinf(max_error)) {
      throw std::invalid_argument(
          "Log maximum error must be positive and finite.");
    }
    if (max_nuts_depth == 0) {
      throw std::invalid_argument("Maximum NUTS depth must be positive.");
    }
    if (min_micro_steps == 0) {
      throw std::invalid_argument("Minimum micro steps must be positive.");
    }
  }

  /**
   * @brief Return the next draw from the sampler.
   *
   * @return The next draw.
   */
  Vec<S> operator()() {
    Vec<S> grad_next;
    theta_ = transition_w(rand_, logp_grad_, inv_mass_, cholesky_mass_,
                          macro_step_size_, max_nuts_depth_, max_step_halvings_,
			  min_micro_steps_, max_error_,
			  std::move(theta_), grad_next, 
			  no_op_adapt_handler_);
    return theta_;
  }

  /**
   * @brief  Return the diagonal of the diagonal inverse mass matrix.
   *
   * @return The diagonal of the inverse mass matrix.
   */
  Vec<S> inverse_mass_matrix_diagonal() const { return inv_mass_; }

  /**
   * @brief Return the macro (largest) step size.
   *
   * @return The largest step size.
   */
  S macro_step_size() const { return macro_step_size_; }

  /**
   * @brief Return the maximum error allowed among Hamiltonians.
   *
   * @return The maximum error allowed among Hamiltonians.
   */
  S max_error() const { return max_error_; }

 private:
  /** The underlying randomizer. */
  Random<S, RNG> rand_;

  /** The target log density/gradient function. */
  const NoExceptLogpGrad<F, S> logp_grad_;

  /** The current position. */
  Vec<S> theta_;

  /** The diagonal of the diagonal inverse mass matrix. */
  const Vec<S> inv_mass_;

  /** The diagonal of the diagonal Cholesky factor of the mass matrix. */
  const Vec<S> cholesky_mass_;

  /** The initial step size. */
  const S macro_step_size_;

  /** The maximum number of doublings in NUTS trajectories. */
  const std::size_t max_nuts_depth_;

  /** The maximum number of halvings of the step size. */
  const std::size_t max_step_halvings_;
  
  /** The minimum number of micro steps per macro step. */
  const std::size_t min_micro_steps_;

  /** The max difference of Hamiltonians along a macro step. */
  const S max_error_;

  /** A handler for adaptation which does nothing. */
  const NoOpHandler no_op_adapt_handler_;
};

}  // namespace nuts
