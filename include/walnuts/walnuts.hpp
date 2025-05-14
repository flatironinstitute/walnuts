#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <utility>

namespace walnuts {

template <typename S> using Vec = Eigen::Matrix<S, Eigen::Dynamic, 1>;

template <typename S>
using Matrix = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;

using Integer = std::int32_t;

enum class Update { Barker, Metropolis };

enum class Direction { Backward, Forward };

template <typename S, class Generator> class Random {
public:
  explicit Random(Generator &rng)
      : rng_(rng), unif_(0.0, 1.0), binary_(0.5), normal_(0.0, 1.0) {}

  inline S uniform_real_01() { return unif_(rng_); }

  inline bool uniform_binary() { return binary_(rng_); }

  inline Vec<S> standard_normal(Integer n) {
    return Vec<S>::NullaryExpr(n, [&](Integer) { return normal_(rng_); });
  }

private:
  Generator &rng_;
  std::uniform_real_distribution<S> unif_;
  std::bernoulli_distribution binary_;
  std::normal_distribution<S> normal_;
};

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
bool reversible(const F &logp_grad_fun, const Vec<S> &inv_mass, S step,
		Integer num_steps, S max_error, const Vec<S> &theta,
		const Vec<S> &rho, const Vec<S> &grad, S logp_next) {
  if (num_steps < 2) {
    return true;
  }
  // ***HEURISTIC***:  just make sure num_steps /= 2 is not stable
  S half_step = step;
  num_steps /= 2;
  step *= 2;
  Vec<S> grad_next(grad);
  Vec<S> theta_next(theta);
  Vec<S> rho_next(-rho);
  S logp_min = logp_next;
  S logp_max = logp_next;
  for (int n = 0; n < num_steps; ++n) {
      rho_next.noalias() = rho_next + half_step * grad_next;
      theta_next.noalias() += step * (inv_mass.array() * rho_next.array()).matrix();
      logp_grad_fun(theta_next, logp_next, grad_next);
      rho_next.noalias() += half_step * grad_next;
      logp_next += logp_momentum(rho_next, inv_mass);
      logp_min = fmin(logp_min, logp_next);
      logp_max = fmax(logp_max, logp_next);
      if (logp_max - logp_min > max_error) {
	return true;
      }
  }
  std::cout << "non-reversible" << std::endl;
  return false;
}  

template <typename S, typename F>
void macro_step(const F &logp_grad_fun, const Vec<S> &inv_mass, S step,
                const Vec<S> &theta, const Vec<S> &rho, const Vec<S> &grad,
                Vec<S> &theta_next, Vec<S> &rho_next, Vec<S> &grad_next,
                S &logp_next, S max_error, bool& irreversible) {
  using std::fmax, std::fmin;
  Vec<S> grad_first;
  S logp_first;
  logp_grad_fun(theta, logp_first, grad_first);
  logp_first += logp_momentum(rho, inv_mass);
  for (int num_steps = 1, halvings = 0; halvings < 10; ++halvings, num_steps *= 2, step *= 0.5) {
    theta_next = theta;
    rho_next = rho;
    grad_next = grad_first;
    S logp_min = logp_first;
    S logp_max = logp_first;
    S half_step = 0.5 * step;
    for (Integer n = 0; n < num_steps; ++n) {
      rho_next.noalias() = rho_next + half_step * grad_next;
      theta_next.noalias() += step * (inv_mass.array() * rho_next.array()).matrix();
      logp_grad_fun(theta_next, logp_next, grad_next);
      rho_next.noalias() += half_step * grad_next;
      logp_next += logp_momentum(rho_next, inv_mass);
      logp_min = fmin(logp_min, logp_next);
      logp_max = fmax(logp_max, logp_next);
      if (logp_max - logp_min > max_error) {
	break;
      }
    }
    if (logp_max - logp_min <= max_error) {
      // if (true) return;
      irreversible = !reversible(logp_grad_fun, inv_mass, step, num_steps,
				 max_error, theta_next, rho_next, grad_next,
				 logp_next);
      return;
    }
  }
  irreversible = true;
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
                   const Vec<S> &inv_mass, S step, S max_error,
                   bool& reversible) {
  Vec<S> theta_next;
  Vec<S> rho_next;
  Vec<S> grad_theta_next;
  S logp_theta_next;
  if constexpr (D == Direction::Forward) {
    macro_step(logp_grad_fun, inv_mass, step, span.theta_fw_, span.rho_fw_,
               span.grad_theta_fw_, theta_next, rho_next, grad_theta_next,
               logp_theta_next, max_error, reversible);
  } else { // Direction::Backward
    macro_step(logp_grad_fun, inv_mass, -step, span.theta_bk_, span.rho_bk_,
               span.grad_theta_bk_, theta_next, rho_next, grad_theta_next,
               logp_theta_next, max_error, reversible);
  }
  return Span<S>(std::move(theta_next), std::move(rho_next),
                 std::move(grad_theta_next), logp_theta_next);
}

template <Direction D, typename S, class F, class Generator>
Span<S> build_span(Random<S, Generator> &rng, const F &logp_grad_fun,
                   const Vec<S> &inv_mass, S step, Integer depth,
                   S max_error, const Span<S> &last_span, bool &uturn_flag) {
  uturn_flag = false;
  if (depth == 0) {
    return build_leaf<D>(logp_grad_fun, last_span, inv_mass, step,
                         max_error, uturn_flag);
  }
  if (uturn_flag) {
    return last_span;  // won't be used
  }
  Span<S> span1 = build_span<D>(rng, logp_grad_fun, inv_mass, step,
                                depth - 1, max_error, last_span, uturn_flag);
  if (uturn_flag) {
    return last_span; // won't be used
  }
  Span<S> span2 = build_span<D>(rng, logp_grad_fun, inv_mass, step,
                                depth - 1, max_error, span1, uturn_flag);
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
                  Vec<S> &&theta, S max_error) {
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
          rng, logp_grad_fun, inv_mass, step, depth, max_error, span_accum, uturn_flag);
      if (uturn_flag)
        break;
      span_accum = combine<Update::Metropolis, Direction::Forward>(
          rng, std::move(span_accum), std::move(span_next), inv_mass,
          uturn_flag);
      if (uturn_flag)
        break;
    } else {
      Span<S> span_next = build_span<Direction::Backward>(
          rng, logp_grad_fun, inv_mass, step, depth, max_error, span_accum, uturn_flag);
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
void walnuts(Generator &generator, const F &logp_grad_fun,
	     const Vec<S> &inv_mass, S step, Integer max_depth, S max_error,
	     const Vec<S> &theta_init, Integer num_draws, H &handler) {
  Random<S, Generator> rng{generator};
  Vec<S> theta = theta_init; // copy once
  handler(0, theta);
  for (Integer n = 1; n < num_draws; ++n) {
    theta = transition(rng, logp_grad_fun, inv_mass, step, max_depth,
                       std::move(theta), max_error);
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

} // namespace nuts
