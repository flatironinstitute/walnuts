#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>

namespace walnuts {

template <typename S> using Vec = Eigen::Matrix<S, Eigen::Dynamic, 1>;

template <typename S>
using Matrix = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;

using Integer = std::int32_t;

enum class Update { Barker, Metropolis };

enum class Direction { Backward, Forward };

enum class CombineResult { Keep, ThrowAway };

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
  S logp_bk_;
  Vec<S> theta_fw_;
  Vec<S> rho_fw_;
  Vec<S> grad_theta_fw_;
  S logp_fw_;
  Vec<S> theta_select_;
  S logp_;

  Span(Vec<S> &&theta, Vec<S> &&rho, Vec<S> &&grad_theta,
       S logp)
    : theta_bk_(theta), rho_bk_(rho), grad_theta_bk_(grad_theta),
      logp_bk_(logp), theta_fw_(theta), rho_fw_(std::move(rho)),
      grad_theta_fw_(std::move(grad_theta)), logp_fw_(logp),
      theta_select_(std::move(theta)), logp_(logp) {}

  Span(Span<S> &&span1, Span<S> &&span2, Vec<S> &&theta_select, S logp)
    : theta_bk_(std::move(span1.theta_bk_)),
      rho_bk_(std::move(span1.rho_bk_)),
      grad_theta_bk_(std::move(span1.grad_theta_bk_)),
      logp_bk_(span1.logp_bk_),
      theta_fw_(std::move(span2.theta_fw_)),
      rho_fw_(std::move(span2.rho_fw_)),
      grad_theta_fw_(std::move(span2.grad_theta_fw_)),
      logp_fw_(span2.logp_fw_), theta_select_(std::move(theta_select)),
      logp_(logp) {}
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
bool within_tolerance(const F &logp_grad_fun, const Vec<S> &inv_mass, S step,
		      Integer num_steps, S max_error, Vec<S> &theta_next,
		      Vec<S> &rho_next, Vec<S> &grad_next, S logp_next) {
  S half_step = 0.5 * step;
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
    // TODO: legal to make this return outside of loop
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
    return true;  // redundant, but avoids constructors
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

template <Direction D, typename S, typename F>
bool macro_step(const F &logp_grad_fun, const Vec<S> &inv_mass, S step,
                const Span<S> &span, Vec<S> &theta_next, Vec<S> &rho_next,
                Vec<S> &grad_next, S &logp_next, S max_error) {
  constexpr bool is_forward = (D == Direction::Forward);
  const Vec<S>& theta =  is_forward ? span.theta_fw_ : span.theta_bk_;
  const Vec<S>& rho = is_forward ? span.rho_fw_ : span.rho_bk_;
  const Vec<S>& grad = is_forward ? span.grad_theta_fw_ : span.grad_theta_bk_;
	S logp = is_forward ? span.logp_fw_ : span.logp_bk_;
  step = is_forward ? step : -step;
  using std::fmax, std::fmin;
  for (int num_steps = 1, halvings = 0; halvings < 10; ++halvings, num_steps *= 2, step *= 0.5) {
    theta_next = theta;
    rho_next = rho;
    grad_next = grad;
    S logp_min = logp;
    S logp_max = logp;
    S half_step = 0.5 * step;
    for (Integer n = 0; n < num_steps && logp_max - logp_min <= max_error; ++n) {
      rho_next.noalias() = rho_next + half_step * grad_next;
      theta_next.noalias() += step * (inv_mass.array() * rho_next.array()).matrix();
      logp_grad_fun(theta_next, logp_next, grad_next);
      rho_next.noalias() += half_step * grad_next;
      logp_next += logp_momentum(rho_next, inv_mass);
      logp_min = fmin(logp_min, logp_next);
      logp_max = fmax(logp_max, logp_next);
    }
    if (logp_max - logp_min <= max_error) {
      return !reversible(logp_grad_fun, inv_mass, step, num_steps, max_error,
                         theta_next, rho_next, grad_next, logp_next);
    }
  }
  return true;
}

template <Direction D, typename S>
inline bool uturn(const Span<S> &span_1, const Span<S> &span_2,
           const Vec<S> &inv_mass) {
  auto&& span_bk = (D == Direction::Forward) ? span_1 : span_2;
  auto&& span_fw = (D == Direction::Forward) ? span_2 : span_1;
  auto scaled_diff =
      (inv_mass.array() * (span_fw.theta_fw_ - span_fw.theta_bk_).array())
          .matrix();
  return span_fw.rho_fw_.dot(scaled_diff) < 0 ||
         span_bk.rho_bk_.dot(scaled_diff) < 0;
}

template <Update U, Direction D, CombineResult R, typename S, class Generator>
std::conditional_t<R == CombineResult::ThrowAway, std::optional<Span<S>>,
                   std::pair<Span<S>, bool>>
combine(Random<S, Generator> &rng, Span<S> &&span_old, Span<S> &&span_new,
        const Vec<S> &inv_mass) {
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

  bool did_uturn = uturn<D>(span_old, span_new, inv_mass);

  if constexpr (R == CombineResult::ThrowAway) {
    if (did_uturn) {
      return std::nullopt;
    }
  }

  Span<S> next = [&]() {
    if constexpr (D == Direction::Forward) {
      return Span<S>(std::move(span_old), std::move(span_new),
                     std::move(selected), logp_total);
    } else { // Direction::Backward
      return Span<S>(std::move(span_new), std::move(span_old),
                     std::move(selected), logp_total);
    }
  }();

  if constexpr (R == CombineResult::ThrowAway) {
    return std::make_optional<Span<S>>(std::move(next));
  } else {
    return std::make_pair(std::move(next), did_uturn);
  }
}

template <Direction D, typename S, class F>
std::optional<Span<S>> build_leaf(const F &logp_grad_fun, const Span<S> &span,
                                  const Vec<S> &inv_mass, S step, S max_error) {
  Vec<S> theta_next;
  Vec<S> rho_next;
  Vec<S> grad_theta_next;
  S logp_theta_next;
  if (macro_step<D>(logp_grad_fun, inv_mass, step, span, theta_next, rho_next,
                    grad_theta_next, logp_theta_next, max_error))
    return std::nullopt;
  return Span<S>(std::move(theta_next), std::move(rho_next),
                 std::move(grad_theta_next), logp_theta_next);
}

template <Direction D, typename S, class F, class Generator>
std::optional<Span<S>> build_span(Random<S, Generator> &rng,
                                  const F &logp_grad_fun,
                                  const Vec<S> &inv_mass, S step, Integer depth,
                                  S max_error, const Span<S> &last_span) {
  if (depth == 0) {
    return build_leaf<D>(logp_grad_fun, last_span, inv_mass, step, max_error);
  }
  auto span1 = build_span<D>(rng, logp_grad_fun, inv_mass, step, depth - 1,
                             max_error, last_span);
  if (!span1) {
    return std::nullopt;
  }

  auto span2 = build_span<D>(rng, logp_grad_fun, inv_mass, step, depth - 1,
                             max_error, *span1);
  if (!span2) {
    return std::nullopt;
  }

  if (uturn<D>(*span1, *span2, inv_mass)) {
    return std::nullopt;
  }

  return combine<Update::Barker, D, CombineResult::ThrowAway>(
      rng, *std::move(span1), *std::move(span2), inv_mass);
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
    const bool go_forward = rng.uniform_binary();
    bool uturn_flag;
    if (go_forward) {
      auto span_next = build_span<Direction::Forward>(
          rng, logp_grad_fun, inv_mass, step, depth, max_error, span_accum);
      if (!span_next)
        break;

      std::tie(span_accum, uturn_flag) =
          combine<Update::Metropolis, Direction::Forward, CombineResult::Keep>(
              rng, std::move(span_accum), *std::move(span_next), inv_mass);

      if (uturn_flag)
        break;
    } else {
      auto span_next = build_span<Direction::Backward>(
          rng, logp_grad_fun, inv_mass, step, depth, max_error, span_accum);
      if (!span_next)
        break;

      std::tie(span_accum, uturn_flag) =
          combine<Update::Metropolis, Direction::Backward, CombineResult::Keep>(
              rng, std::move(span_accum), *std::move(span_next), inv_mass);

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

} // namespace walnuts
