#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <utility>

#include "util.hpp"

namespace nuts {

template <typename S> class Span {
public:
  Vec<S> theta_bk_;
  Vec<S> rho_bk_;
  Vec<S> grad_theta_bk_;
  Vec<S> theta_fw_;
  Vec<S> rho_fw_;
  Vec<S> grad_theta_fw_;
  Vec<S> theta_select_;
  S logp_;

  Span(Vec<S> &&theta, Vec<S> &&rho, Vec<S> &&grad_theta,
       S logp)
      : theta_bk_(theta), // copy duplicates
        rho_bk_(rho), grad_theta_bk_(grad_theta), theta_fw_(theta),
        rho_fw_(std::move(rho)), // move once after copy
        grad_theta_fw_(std::move(grad_theta)), theta_select_(std::move(theta)),
        logp_(logp) {}

  Span(Span<S> &&span1, Span<S> &&span2, Vec<S> &&theta_select, S logp)
      : theta_bk_(std::move(span1.theta_bk_)),
        rho_bk_(std::move(span1.rho_bk_)),
        grad_theta_bk_(std::move(span1.grad_theta_bk_)),
        theta_fw_(std::move(span2.theta_fw_)),
        rho_fw_(std::move(span2.rho_fw_)),
        grad_theta_fw_(std::move(span2.grad_theta_fw_)),
        theta_select_(std::move(theta_select)), logp_(logp) {}
};

template <typename S> S log_sum_exp(const S &x1, const S &x2) {
  using std::fmax, std::log, std::exp;
  S m = fmax(x1, x2);
  return m + log(exp(x1 - m) + exp(x2 - m));
}

template <typename S> S log_sum_exp(const Vec<S> &x) {
  using std::log;
  S m = x.maxCoeff();
  return m + log((x.array() - m).exp().sum());
}

template <typename S>
S logp_momentum(const Vec<S> &rho, const Vec<S> &inv_mass) {
  return -0.5 * rho.dot(inv_mass.cwiseProduct(rho));
}

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

template <typename S>
bool uturn(const Span<S> &span_bk, const Span<S> &span_fw,
           const Vec<S> &inv_mass) {
  auto scaled_diff =
      (inv_mass.array() * (span_fw.theta_fw_ - span_fw.theta_bk_).array())
          .matrix();
  return span_fw.rho_fw_.dot(scaled_diff) < 0 ||
         span_bk.rho_bk_.dot(scaled_diff) < 0;
}

template <Update U, Direction D, typename S, class Generator>
Span<S> combine(Random<S, Generator> &rng, Span<S> &&span_old,
                Span<S> &&span_new, const Vec<S> &inv_mass, bool &uturn_flag) {
  using std::log;
  S logp_total = log_sum_exp(span_old.logp_, span_new.logp_);
  S log_denominator;
  if constexpr (U == Update::Metropolis) {
    log_denominator = span_new.logp_;
  } else { // Update::Barker
    log_denominator = logp_total;
  }
  S update_logprob = span_new.logp_ - log_denominator;
  bool update = log(rng.uniform_real_01()) < update_logprob;
  auto &selected = update ? span_new.theta_select_ : span_old.theta_select_;
  if constexpr (D == Direction::Forward) {
    uturn_flag = uturn(span_old, span_new, inv_mass);
    return Span<S>(std::move(span_old), std::move(span_new),
                   std::move(selected), logp_total);
  } else { // Direction::Backward
    uturn_flag = uturn(span_new, span_old, inv_mass);
    return Span<S>(std::move(span_new), std::move(span_old),
                   std::move(selected), logp_total);
  }
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
  } else { // Direction::Backward
    leapfrog(logp_grad_fun, inv_mass, -step, span.theta_bk_, span.rho_bk_,
             span.grad_theta_bk_, theta_next, rho_next, grad_theta_next,
             logp_theta_next);
  }
  return Span<S>(std::move(theta_next), std::move(rho_next),
                 std::move(grad_theta_next), logp_theta_next);
}

template <Direction D, typename S, class F, class Generator>
Span<S> build_span(Random<S, Generator> &rng, const F &logp_grad_fun,
                   const Vec<S> &inv_mass, S step, Integer depth,
                   const Span<S> &last_span, bool &uturn_flag) {
  uturn_flag = false;
  if (depth == 0) {
    return build_leaf<D>(logp_grad_fun, last_span, inv_mass, step);
  }
  Span<S> span1 = build_span<D>(rng, logp_grad_fun, inv_mass, step, depth - 1,
                                last_span, uturn_flag);
  if (uturn_flag) {
    return last_span; // won't be used
  }
  Span<S> span2 = build_span<D>(rng, logp_grad_fun, inv_mass, step, depth - 1,
                                span1, uturn_flag);
  if (uturn_flag) {
    return last_span; // won't be used
  }
  if constexpr (D == Direction::Forward) {
    if (uturn(span1, span2, inv_mass)) {
      uturn_flag = true;
      return last_span; // won't be used
    }
  } else { // Direction::Backward
    if (uturn(span2, span1, inv_mass)) {
      uturn_flag = true;
      return last_span; // won't be used
    }
  }
  return combine<Update::Barker, D>(rng, std::move(span1), std::move(span2),
                                    inv_mass, uturn_flag);
}

template <typename S, class F, class Generator>
Vec<S> transition(Random<S, Generator> &rng, const F &logp_grad_fun,
                  const Vec<S> &inv_mass, S step, Integer max_depth,
                  Vec<S> &&theta) {
  Vec<S> rho = rng.standard_normal(theta.size());
  Vec<S> grad(theta.size());
  S logp;
  logp_grad_fun(theta, logp, grad);
  logp += logp_momentum(rho, inv_mass);
  Span<S> span_accum(std::move(theta), std::move(rho), std::move(grad), logp);
  for (Integer depth = 0; depth < max_depth; ++depth) {
    bool go_forward = rng.uniform_binary();
    bool uturn_flag;
    if (go_forward) {
      Span<S> span_next = build_span<Direction::Forward>(
          rng, logp_grad_fun, inv_mass, step, depth, span_accum, uturn_flag);
      if (uturn_flag)
        break;
      span_accum = combine<Update::Metropolis, Direction::Forward>(
          rng, std::move(span_accum), std::move(span_next), inv_mass,
          uturn_flag);
      if (uturn_flag)
        break;
    } else {
      Span<S> span_next = build_span<Direction::Backward>(
          rng, logp_grad_fun, inv_mass, step, depth, span_accum, uturn_flag);
      if (uturn_flag)
        break;
      span_accum = combine<Update::Metropolis, Direction::Backward>(
          rng, std::move(span_accum), std::move(span_next), inv_mass,
          uturn_flag);
      if (uturn_flag)
        break;
    }
  }
  return std::move(span_accum.theta_select_);
}

template <typename S, class F, class Generator, class H>
void nuts(Generator &generator, const F &logp_grad_fun, const Vec<S> &inv_mass,
          S step, Integer max_depth, const Vec<S> &theta_init,
          Integer num_draws, H &handler) {
  Random<S, Generator> rng{generator};
  Vec<S> theta = theta_init; // copy once
  handler(0, theta);
  for (Integer n = 1; n < num_draws; ++n) {
    theta = transition(rng, logp_grad_fun, inv_mass, step, max_depth,
                       std::move(theta));
    handler(n, theta);
  }
}

template <typename S, class F, class Generator>
void nuts(Generator &generator, const F &logp_grad_fun, const Vec<S> &inv_mass,
          S step, Integer max_depth, const Vec<S> &theta_init,
          Matrix<S> &sample) {
  auto handler = [&sample](Integer n, const Vec<S> &v) { sample.col(n) = v; };
  nuts(generator, logp_grad_fun, inv_mass, step, max_depth, theta_init,
       sample.cols(), handler);
}

} // namespace nuts
