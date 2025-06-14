#ifndef NUTS_WALNUTS_HPP
#define NUTS_WALNUTS_HPP

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
class SpanW {
 public:
  SpanW(Vec<S> &&theta, Vec<S> &&rho, Vec<S> &&grad_theta, S logp)
      : theta_bk_(theta),
        rho_bk_(rho),
        grad_theta_bk_(grad_theta),
        logp_bk_(logp),
        theta_fw_(theta),
        rho_fw_(std::move(rho)),
        grad_theta_fw_(std::move(grad_theta)),
        logp_fw_(logp),
        theta_select_(std::move(theta)),
        logp_(logp) {}

  SpanW(SpanW<S> &&span1, SpanW<S> &&span2, Vec<S> &&theta_select, S logp)
      : theta_bk_(std::move(span1.theta_bk_)),
        rho_bk_(std::move(span1.rho_bk_)),
        grad_theta_bk_(std::move(span1.grad_theta_bk_)),
        logp_bk_(span1.logp_bk_),
        theta_fw_(std::move(span2.theta_fw_)),
        rho_fw_(std::move(span2.rho_fw_)),
        grad_theta_fw_(std::move(span2.grad_theta_fw_)),
        logp_fw_(span2.logp_fw_),
        theta_select_(std::move(theta_select)),
        logp_(logp) {}

  Vec<S> theta_bk_;
  Vec<S> rho_bk_;
  Vec<S> grad_theta_bk_;
  S logp_bk_;
  Vec<S> theta_fw_;
  Vec<S> rho_fw_;
  Vec<S> grad_theta_fw_;
  S logp_fw_;
  Vec<S> theta_select_;
  S logp_;
};

template <typename S, typename F>
bool within_tolerance(const F &logp_grad_fun, const Vec<S> &inv_mass, S step,
                      Integer num_steps, S max_error, Vec<S> &theta_next,
                      Vec<S> &rho_next, Vec<S> &grad_next, S logp_next) {
  S half_step = 0.5 * step;
  S logp_min = logp_next;
  S logp_max = logp_next;
  for (int n = 0; n < num_steps; ++n) {
    rho_next.noalias() = rho_next + half_step * grad_next;
    theta_next.noalias() +=
        step * (inv_mass.array() * rho_next.array()).matrix();
    logp_grad_fun(theta_next, logp_next, grad_next);
    rho_next.noalias() += half_step * grad_next;
    logp_next += logp_momentum(rho_next, inv_mass);
    logp_min = fmin(logp_min, logp_next);
    logp_max = fmax(logp_max, logp_next);
    // TODO: eval alternative with this test outside of loop
    if (logp_max - logp_min > max_error) {
      return false;
    }
  }
  return true;
}

template <typename S, typename F>
bool reversible(const F &logp_grad_fun, const Vec<S> &inv_mass, S step,
                Integer num_steps, S max_error, const Vec<S> &theta,
                const Vec<S> &rho, const Vec<S> &grad, S logp_next) {
  if (num_steps == 1) {
    return true;
  }
  Vec<S> theta_next(grad.size());
  Vec<S> rho_next(rho.size());
  Vec<S> grad_next(grad.size());
  while (num_steps >= 2) {
    theta_next = theta;
    rho_next = -rho;
    grad_next = grad;
    num_steps /= 2;
    step *= 2;
    if (within_tolerance(logp_grad_fun, inv_mass, step, num_steps, max_error,
                         theta_next, rho_next, grad_next, logp_next)) {
      return false;
    }
  }
  return true;
}

template <Direction D, typename S, typename F, class C>
bool macro_step(const F &logp_grad_fun, const Vec<S> &inv_mass, S step,
                const SpanW<S> &span, Vec<S> &theta_next, Vec<S> &rho_next,
                Vec<S> &grad_next, S &logp_next, S max_error,
                C& adapt_handler) {
  constexpr bool is_forward = (D == Direction::Forward);
  const Vec<S> &theta = is_forward ? span.theta_fw_ : span.theta_bk_;
  const Vec<S> &rho = is_forward ? span.rho_fw_ : span.rho_bk_;
  const Vec<S> &grad = is_forward ? span.grad_theta_fw_ : span.grad_theta_bk_;
  S logp = is_forward ? span.logp_fw_ : span.logp_bk_;
  step = is_forward ? step : -step;
  using std::fmax, std::fmin;
  for (int num_steps = 1, halvings = 0; halvings < 10;
       ++halvings, num_steps *= 2, step *= 0.5) {
    theta_next = theta;
    rho_next = rho;
    grad_next = grad;
    S logp_min = logp;
    S logp_max = logp;
    S half_step = 0.5 * step;
    for (Integer n = 0; n < num_steps && logp_max - logp_min <= max_error;
         ++n) {
      rho_next.noalias() = rho_next + half_step * grad_next;
      theta_next.noalias()
          += step * (inv_mass.array() * rho_next.array()).matrix();
      logp_grad_fun(theta_next, logp_next, grad_next);
      rho_next.noalias() += half_step * grad_next;
      logp_next += logp_momentum(rho_next, inv_mass);
      logp_min = fmin(logp_min, logp_next);
      logp_max = fmax(logp_max, logp_next);
      if (num_steps == 1) {
        S min_accept = std::exp(logp_min - logp_max);
        adapt_handler(min_accept);
      }
    }
    if (logp_max - logp_min <= max_error) {
      return !reversible(logp_grad_fun, inv_mass, step, num_steps, max_error,
                         theta_next, rho_next, grad_next, logp_next);
    }
  }
  return true;
}

template <Update U, Direction D, typename S, class Generator>
SpanW<S> combine(Random<S, Generator> &rng, SpanW<S> &&span_old,
                 SpanW<S> &&span_new) {
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
  return SpanW<S>(std::move(span_bk), std::move(span_fw), std::move(selected),
                  logp_total);
}

template <Direction D, typename S, class F, class C>
std::optional<SpanW<S>> build_leaf(const F &logp_grad_fun, const SpanW<S> &span,
                                   const Vec<S> &inv_mass, S step,
                                   S max_error, C& adapt_handler) {
  Vec<S> theta_next;
  Vec<S> rho_next;
  Vec<S> grad_theta_next;
  S logp_theta_next;
  if (macro_step<D>(logp_grad_fun, inv_mass, step, span, theta_next, rho_next,
                    grad_theta_next, logp_theta_next, max_error,
                    adapt_handler)) {
    return std::nullopt;
  }
  return SpanW<S>(std::move(theta_next), std::move(rho_next),
                  std::move(grad_theta_next), logp_theta_next);
}

template <Direction D, typename S, class F, class Generator, class C>
std::optional<SpanW<S>> build_span(Random<S, Generator> &rng,
                                   const F &logp_grad_fun,
                                   const Vec<S> &inv_mass, S step,
                                   Integer depth, S max_error,
                                   const SpanW<S> &last_span,
                                   C& adapt_handler) {
  if (depth == 0) {
    return build_leaf<D>(logp_grad_fun, last_span, inv_mass, step, max_error,
                         adapt_handler);
  }
  auto maybe_subspan1 = build_span<D>(rng, logp_grad_fun, inv_mass, step,
                                      depth - 1, max_error, last_span,
                                      adapt_handler);
  if (!maybe_subspan1) {
    return std::nullopt;
  }
  auto maybe_subspan2 = build_span<D>(rng, logp_grad_fun, inv_mass, step,
                                      depth - 1, max_error, *maybe_subspan1,
                                      adapt_handler);
  if (!maybe_subspan2) {
    return std::nullopt;
  }
  if (uturn<D>(*maybe_subspan1, *maybe_subspan2, inv_mass)) {
    return std::nullopt;
  }
  return std::make_optional(combine<Update::Barker, D>(
      rng, std::move(*maybe_subspan1), std::move(*maybe_subspan2)));
}

template <typename S, class F, class Generator, class C>
Vec<S> transition_w(Random<S, Generator> &rng, const F &logp_grad_fun,
                    const Vec<S> &inv_mass, const Vec<S> &chol_mass, S step,
                    Integer max_depth, Vec<S> &&theta, S max_error,
                    C& adapt_handler) {
  Vec<S> rho = rng.standard_normal(theta.size()).cwiseProduct(chol_mass);
  Vec<S> grad(theta.size());
  S logp;
  logp_grad_fun(theta, logp, grad);
  logp += logp_momentum(rho, inv_mass);
  SpanW<S> span_accum(std::move(theta), std::move(rho), std::move(grad), logp);
  for (Integer depth = 0; depth < max_depth; ++depth) {
    const bool go_forward = rng.uniform_binary();
    if (go_forward) {
      constexpr Direction D = Direction::Forward;
      auto maybe_next_span = build_span<D>(rng, logp_grad_fun, inv_mass, step,
                                           depth, max_error, span_accum,
                                           adapt_handler);
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
      auto span_next = build_span<D>(rng, logp_grad_fun, inv_mass, step, depth,
                                     max_error, span_accum, adapt_handler);
      if (!span_next) {
        break;
      }
      bool combined_uturn = uturn<D>(span_accum, *span_next, inv_mass);
      span_accum = combine<Update::Metropolis, D>(rng, std::move(span_accum),
                                                  std::move(*span_next));
      if (combined_uturn) {
        break;
      }
    }
  }
  return std::move(span_accum.theta_select_);
}

class NoOpHandler {
 public:
  template <typename T>
  void operator()(const T& x) const noexcept { }
};

template <typename S, class F, class Generator, class H>
void walnuts(Generator &generator, const F &logp_grad_fun,
             const Vec<S> &inv_mass, S step, Integer max_depth, S max_error,
             const Vec<S> &theta_init, Integer num_draws, H &handler) {
  NoOpHandler adapt_handler;
  Random<S, Generator> rng{generator};
  Vec<S> chol_mass = inv_mass.array().sqrt().inverse().matrix();
  Vec<S> theta = theta_init;
  handler(0, theta);
  for (Integer n = 1; n < num_draws; ++n) {
    theta = transition_w(rng, logp_grad_fun, inv_mass, chol_mass, step,
                         max_depth, std::move(theta), max_error, adapt_handler);
    handler(n, theta);
  }
}

template <typename S, class F, class Generator>
void walnuts(Generator &generator, const F &logp_grad_fun,
             const Vec<S> &inv_mass, S step, Integer max_depth, S max_error,
             const Vec<S> &theta_init, Matrix<S> &sample) {
  auto handler = [&sample](Integer n, const Vec<S> &v) { sample.col(n) = v; };
  walnuts(generator, logp_grad_fun, inv_mass, step, max_depth, max_error,
          theta_init, sample.cols(), handler);
}


template <class F, typename S, class RNG>
class WalnutsSampler {
 public:
  WalnutsSampler(Random<S, RNG>& rand,
		 F& logp_grad,
                 const Vec<S>& theta,
                 const Vec<S>& inv_mass,
                 S macro_step_size,
                 Integer max_nuts_depth,
                 S log_max_error):
      rand_(rand), logp_grad_(logp_grad), theta_(theta), inv_mass_(inv_mass),
      cholesky_mass_(inv_mass.array().sqrt().inverse().matrix()),
      macro_step_size_(macro_step_size), max_nuts_depth_(max_nuts_depth),
      log_max_error_(log_max_error), no_op_adapt_handler_()
  {
    std::cout << "inv_mass = " << inv_mass.transpose() << std::endl;
    std::cout << "macro_step_size = " << macro_step_size << std::endl;
  }

  Vec<S> operator()() {
    theta_ = transition_w(rand_, logp_grad_, inv_mass_, cholesky_mass_,
			  macro_step_size_, max_nuts_depth_, std::move(theta_),
                          log_max_error_, no_op_adapt_handler_);
    return theta_;
  }

 private:
  Random<S, RNG> rand_;
  F& logp_grad_;
  Vec<S> theta_;
  const Vec<S> inv_mass_;
  const Vec<S> cholesky_mass_;
  const S macro_step_size_;
  const Integer max_nuts_depth_;
  const S log_max_error_;
  const NoOpHandler no_op_adapt_handler_;
};

template <class F, typename S, class RNG>
WalnutsSampler(RNG&, F&, const Vec<S>&, const Vec<S>&, S, Integer, S)
  -> WalnutsSampler<F, S, RNG>;

}  // namespace nuts
#endif
