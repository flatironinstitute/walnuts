#include <algorithm>
#include <cmath>
#include <utility>
#include <random>
#include <Eigen/Dense>

#include "Eigen/src/Core/Matrix.h"
#include "arena_matrix.hpp"
#include "memory.hpp"

namespace nuts {

template <typename S>
using Vec = arena_matrix<Eigen::Matrix<S, Eigen::Dynamic, 1>>;
static thread_local arena_allocator<double, nuts::arena_alloc> default_arena_alloc;

template <typename S>
using Matrix = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;

using Integer = std::int32_t;

template <typename S, class Generator>
class Random {
 public:
  Random(Generator& rng): rng_(rng), unif_(0.0, 1.0), binary_(0.5), normal_(0.0, 1.0) { }

  S uniform_real_01() {
    return unif_(rng_);
  }

  bool uniform_binary() {
    return binary_(rng_);
  }

  Vec<S> standard_normal(Integer n) {
    return Vec<S>(default_arena_alloc, Vec<S>::Base::NullaryExpr(n, [&](Integer) { return normal_(rng_); }));
  }
 private:
  Generator& rng_;
  std::uniform_real_distribution<S> unif_;
  std::bernoulli_distribution binary_;
  std::normal_distribution<S> normal_;
};


template <typename S>
class Span {
 public:
  Vec<S> theta_bk_;
  Vec<S> rho_bk_;
  Vec<S> grad_theta_bk_;
  Vec<S> theta_fw_;
  Vec<S> rho_fw_;
  Vec<S> grad_theta_fw_;
  Vec<S> theta_select_;
  S logp_;

  Span(Vec<S>&& theta,
       Vec<S>&& rho,
       Vec<S>&& grad_theta,
       S logp)
      : theta_bk_(default_arena_alloc, theta),  // copy duplicates
        rho_bk_(default_arena_alloc, rho),
        grad_theta_bk_(default_arena_alloc, grad_theta),
        theta_fw_(default_arena_alloc, theta),
        rho_fw_(std::move(rho)),  // move once after copy
        grad_theta_fw_(std::move(grad_theta)),
        theta_select_(std::move(theta)),
        logp_(logp)
  {}

  Span(Span<S>&& span1,
       Span<S>&& span2,
       Vec<S>&& theta_select,
       S logp)
      : theta_bk_(std::move(span1.theta_bk_)),
        rho_bk_(std::move(span1.rho_bk_)),
        grad_theta_bk_(std::move(span1.grad_theta_bk_)),
        theta_fw_(std::move(span2.theta_fw_)),
        rho_fw_(std::move(span2.rho_fw_)),
        grad_theta_fw_(std::move(span2.grad_theta_fw_)),
        theta_select_(std::move(theta_select)),
        logp_(logp)
  {}
};

template <typename S>
S log_sum_exp(const S& x1, const S& x2) {
  using std::fmax;
  using std::log;
  using std::exp;
  S m = fmax(x1, x2);
  return m + log(exp(x1 - m) + exp(x2 - m));
}

template <typename S>
S log_sum_exp(const Vec<S>& x) {
  using std::log;
  S m = x.maxCoeff();
  return m + log((x.array() - m).exp().sum());
}

template <typename S>
S logp_momentum(const Vec<S>& rho,
                const Vec<S>& inv_mass) {
  return -0.5 * (inv_mass.array() * rho.array().square()).sum();
}

template <typename S, typename F>
void leapfrog(const F& logp_grad_fun,
              const Vec<S>& inv_mass,
              S step,
              const Vec<S>& theta,
              const Vec<S>& rho,
              const Vec<S>& grad,
              Vec<S>& theta_next,
              Vec<S>& rho_next,
              Vec<S>& grad_next,
              S& logp_next) {
  S half_step = 0.5 * step;
  rho_next = rho + half_step * grad;
  theta_next = theta + step * (inv_mass.array() * rho_next.array()).matrix();
  logp_grad_fun(theta_next, logp_next, grad_next);
  rho_next.noalias() += half_step * grad_next;
  logp_next += logp_momentum(rho_next, inv_mass);
}

template <typename S, class F>
bool stable(const F& logp_grad_fun,
            const Vec<S>& inv_mass,
            S step,
            Integer L,
            S max_energy_error,
            const Vec<S>& theta,
            const Vec<S>& rho,
            const Vec<S>& grad,
            S logp,
            Vec<S>& theta_next,
            Vec<S>& rho_next,
            Vec<S>& grad_next,
            S& logp_next) {
  using std::fmax;
  using std::fmin;
  S logp_min = logp;
  S logp_max = logp;
  theta_next = theta;
  rho_next = rho;
  grad_next = grad;
  for (Integer ell = 0; ell < L; ++ell) {
    S logp_next;
    leapfrog(logp_grad_fun, inv_mass, step, theta_next, rho_next,
             grad_next, theta_next, rho_next, grad_next, logp_next);
    logp_min = fmin(logp_min, logp_next);
    logp_max = fmax(logp_max, logp_next);
    if (logp_max - logp_min > max_energy_error)
      return false;
  }
  return true;
}


template <typename S, class F>
Integer stable_num_steps(const F& logp_grad_fun,
                         const Vec<S>& inv_mass,
                         S macro_step,
                         S max_energy_error,
                         const Vec<S>& theta,
                         const Vec<S>& rho,
                         const Vec<S>& grad,
                         S logp,
                         Vec<S>& theta_next,
                         Vec<S>& rho_next,
                         Vec<S>& grad_next,
                         S& logp_next) {
  S step = macro_step;
  Integer L = 1;
  for (Integer n = 0; n < 10; ++n) {
    if (stable(logp_grad_fun, theta, rho, grad, logp, step, L, max_energy_error,
               theta_next, rho_next, grad_next, logp_next)) {
      return L;
    }
    step /= 2;
    L *= 2;
  }
  return -1;
}

template <typename S>
bool uturn(const Span<S>& span_bk,
           const Span<S>& span_fw,
           const Vec<S>& inv_mass) {
  auto scaled_diff = (inv_mass.array() * (span_fw.theta_fw_ - span_fw.theta_bk_).array()).matrix();
  return span_fw.rho_fw_.dot(scaled_diff) < 0 || span_bk.rho_bk_.dot(scaled_diff) < 0;
}

template <bool Progressive, bool Forward, typename S, class Generator>
Span<S> combine(Random<S, Generator>& rng,
                Span<S>&& span_old,
                Span<S>&& span_new,
                const Vec<S>& inv_mass,
                bool& uturn_flag) {
  using std::log;
  S logp_total = log_sum_exp(span_old.logp_, span_new.logp_);
  S log_denominator;
  if constexpr (Progressive) {
    log_denominator = span_new.logp_;
  } else {
    log_denominator = logp_total;
  }
  S update_logprob = span_new.logp_ - log_denominator;
  bool update = log(rng.uniform_real_01()) < update_logprob;
  auto& selected = update ? span_new.theta_select_ : span_old.theta_select_;
  if constexpr (Forward) {
    uturn_flag = uturn(span_old, span_new, inv_mass);
    return Span<S>(std::move(span_old), std::move(span_new), std::move(selected), logp_total);
  } else {
    uturn_flag = uturn(span_new, span_old, inv_mass);
    return Span<S>(std::move(span_new), std::move(span_old), std::move(selected), logp_total);
  }
}

template <bool Forward, typename S, class F>
Span<S> build_leaf(const F& logp_grad_fun,
                   const Span<S>& span,
                   const Vec<S>& inv_mass,
                   S step,
                   bool& uturn_flag) {
    Vec<S> theta_next(default_arena_alloc);
    Vec<S> rho_next(default_arena_alloc);
    Vec<S> grad_theta_next(default_arena_alloc);
    S logp_theta_next;
    if constexpr (Forward) {
      leapfrog(logp_grad_fun, inv_mass, step, span.theta_fw_, span.rho_fw_, span.grad_theta_fw_,
               theta_next, rho_next, grad_theta_next,
               logp_theta_next);
    } else {
      leapfrog(logp_grad_fun, inv_mass, -step, span.theta_bk_, span.rho_bk_, span.grad_theta_bk_,
               theta_next, rho_next, grad_theta_next,
               logp_theta_next);
    }
    uturn_flag = false;
    return Span<S>(std::move(theta_next), std::move(rho_next), std::move(grad_theta_next), logp_theta_next);
}

template <bool Forward, typename S, class F, class Generator>
Span<S> build_span(Random<S, Generator>& rng,
                   const F& logp_grad_fun,
                   const Vec<S>& inv_mass,
                   S step,
                   Integer depth,
                   const Span<S>& last_span,
                   bool& uturn_flag) {
  if (depth == 0) {
    return build_leaf<Forward>(logp_grad_fun, last_span, inv_mass, step, uturn_flag);
  }
  Span<S> span1 = build_span<Forward>(rng, logp_grad_fun, inv_mass, step,
                                      depth - 1, last_span, uturn_flag);
  if (uturn_flag) {
    return last_span;  // won't be used
  }
  Span<S> span2 = build_span<Forward>(rng, logp_grad_fun, inv_mass, step,
                                      depth - 1, span1, uturn_flag);
  if (uturn_flag) {
    return last_span; // won't be used
  }
  if constexpr (Forward) {
    if (uturn(span1, span2, inv_mass)) {
      uturn_flag = true;
      return last_span;  // won't be used
    }
  } else {
    if (uturn(span2, span1, inv_mass)) {
      uturn_flag = true;
      return last_span;  // won't be used
    }
  }
  return combine<false, Forward>(rng, std::move(span1), std::move(span2), inv_mass, uturn_flag);
}

template <typename S, class F, typename V, class Generator>
void transition(Random<S, Generator>& rng,
                const F& logp_grad_fun,
                const Vec<S>& inv_mass,
                S step,
                Integer max_depth,
                Vec<S>&& theta,
                V theta_next) {
  Vec<S> rho = rng.standard_normal(theta.size());
  S logp;
  Vec<S> grad(default_arena_alloc,theta.size());
  logp_grad_fun(theta, logp, grad);
  logp += logp_momentum(rho, inv_mass);
  Span<S> span_accum(std::move(theta), std::move(rho), std::move(grad), logp);
  for (Integer depth = 0; depth < max_depth; ++depth) {
    bool go_forward = rng.uniform_binary();
    bool uturn_flag;
    if (go_forward) {
      Span<S> span_next = build_span<true>(rng, logp_grad_fun, inv_mass, step, depth, span_accum, uturn_flag);
      if (uturn_flag) break;
      span_accum = combine<true, true>(rng, std::move(span_accum), std::move(span_next), inv_mass, uturn_flag);
      if (uturn_flag) break;
    } else {
      Span<S> span_next = build_span<false>(rng, logp_grad_fun, inv_mass, step, depth, span_accum, uturn_flag);
      if (uturn_flag) break;
      span_accum = combine<true, false>(rng, std::move(span_accum), std::move(span_next), inv_mass, uturn_flag);
      if (uturn_flag) break;
    }
  }
  theta_next = span_accum.theta_select_;
}

template <typename S, class F, class Generator>
void nuts(Generator& generator,
          const F& logp_grad_fun,
          const Eigen::VectorX<S>& inv_mass,
          S step,
          Integer max_depth,
          const Eigen::VectorX<S>& theta,
          Matrix<S>& sample) {
  Random<S, Generator> rng{generator};
  Integer num_draws = sample.cols();
  if (num_draws == 0) return;
  sample.col(0) = theta;
  auto inv_arena = Vec<S>(default_arena_alloc, inv_mass);

  for (Integer n = 1; n < num_draws; ++n) {
    scoped_allocator scoped_arena_alloc(default_arena_alloc);

    transition(rng, logp_grad_fun, inv_arena, step, max_depth,
               Vec<S>(default_arena_alloc,sample.col(n - 1)), sample.col(n));

  }
}

} // namespace nuts
