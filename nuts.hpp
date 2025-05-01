#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <Eigen/Dense>

// TODO(carpenter): distributions either (a) thread localize or (b) plumb through like RNG

template <typename S>
using Vec = Eigen::Matrix<S, Eigen::Dynamic, 1>;

template <typename S>
using Matrix = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;



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

  // assumes _bk and _fw ae distinct; see other ctor
  Span(const Vec<S> theta_bk,
       const Vec<S> rho_bk,
       const Vec<S> grad_theta_bk,
       const Vec<S> theta_fw,
       const Vec<S> rho_fw,
       const Vec<S> grad_theta_fw,
       const Vec<S> theta_select,
       S logp)
      : theta_bk_(std::move(theta_bk)),
        rho_bk_(std::move(rho_bk)),
        grad_theta_bk_(std::move(grad_theta_bk)),
        theta_fw_(std::move(theta_fw)),
        rho_fw_(std::move(rho_fw)),
        grad_theta_fw_(std::move(grad_theta_fw)),
        theta_select_(std::move(theta_select)),
        logp_(logp)
  {}

  Span(const Vec<S> theta,
       const Vec<S> rho,
       const Vec<S> grad_theta,
       S logp)
      : theta_bk_(theta),  // copy duplicates
        rho_bk_(rho),
        grad_theta_bk_(grad_theta),
        theta_fw_(theta),
        rho_fw_(std::move(rho)),  // move once after copy
        grad_theta_fw_(std::move(grad_theta)),
        theta_select_(std::move(theta)),
        logp_(logp)
  {}

  Span(Span<S>&& span1,
       Span<S>&& span2,
       Vec<S>&& theta_select,
       S logp)
      : theta_bk_(std::move(span1.theta_bk_)), rho_bk_(std::move(span1.rho_bk_)),
        grad_theta_bk_(std::move(span1.grad_theta_bk_)), theta_fw_(std::move(span2.theta_fw_)),
        rho_fw_(std::move(span2.rho_fw_)), grad_theta_fw_(std::move(span2.grad_theta_fw_)),
        theta_select_(std::move(theta_select)), logp_(logp) {}
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


template <typename S, class F>
S potential(const Vec<S>& theta,
            const F& logp_grad_fun) {
  Vec<S> grad;
  S logp;
  logp_grad_fun(theta, logp, grad);
  return -logp;
}


template <typename S>
S kinetic(const Vec<S>& rho,
          const Vec<S>& inv_mass) {
  return 0.5 * (inv_mass.array() * rho.array().square()).sum();
}


template <typename S, class F>
S hamiltonian(const Vec<S>& theta,
              const Vec<S>& rho,
              const Vec<S>& inv_mass,
              const F& logp_grad_fun) {
  return potential(theta, logp_grad_fun) + kinetic(rho, inv_mass);
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
  logp_next -= kinetic(rho_next, inv_mass);
}

template <typename S>
bool uturn(const Vec<S>& theta_bk,
           const Vec<S>& rho_bk,
           const Vec<S>& theta_fw,
           const Vec<S>& rho_fw,
           const Vec<S>& inv_mass) {
  auto scaled_diff = (inv_mass.array() * (theta_fw - theta_bk).array()).matrix();
  return rho_fw.dot(scaled_diff) < 0 || rho_bk.dot(scaled_diff) < 0;
}


template <bool Progressive, typename S, typename RNG>
Span<S> combine_bk(RNG& rng,
                Span<S>&& span1,
                Span<S>&& span2,
                const Vec<S>& inv_mass,
                bool& uturn_flag) {
  using std::log;
  static std::uniform_real_distribution<> unif(0.0, 1.0);
  S logp12 = log_sum_exp(span1.logp_, span2.logp_);

  S log_denominator;
  if constexpr (Progressive) {
    log_denominator = span2.logp_;
  } else {
    log_denominator = logp12;
  }
  S update_logprob = span1.logp_ - log_denominator;
  bool update = log(unif(rng)) < update_logprob;
  auto selected = update ? span1.theta_select_ : span2.theta_select_;
  uturn_flag = uturn(span1.theta_bk_, span1.rho_bk_, span2.theta_fw_,
                     span2.rho_fw_, inv_mass);
  return Span<S>(std::move(span1), std::move(span2), std::move(selected), logp12);
}

template <bool Progressive, typename S, typename RNG>
Span<S> combine_fw(RNG& rng,
                Span<S>&& span1,
                Span<S>&& span2,
                const Vec<S>& inv_mass,
                bool& uturn_flag) {
  using std::log;
  static std::uniform_real_distribution<> unif(0.0, 1.0);
  S logp12 = log_sum_exp(span1.logp_, span2.logp_);

  S log_denominator;
  if constexpr (Progressive) {
    log_denominator = span1.logp_;
  } else {
    log_denominator = logp12;
  }

  S update_logprob = span2.logp_ - log_denominator;
  bool update = log(unif(rng)) < update_logprob;
  auto selected = update ? span2.theta_select_ : span1.theta_select_;
  uturn_flag = uturn(span1.theta_bk_, span1.rho_bk_, span2.theta_fw_, span2.rho_fw_, inv_mass);
  return Span<S>(std::move(span1), std::move(span2), std::move(selected), logp12);
}

// TODO(carpenter): unify build_span_fw() and build_span_bk() into build_span()
template <typename S, class RNG, class F>
Span<S> build_span_bk(RNG& rng,
                      const F& logp_grad_fun,
                      const Vec<S>& inv_mass,
                      S step,
                      int depth,
                      const Span<S>& last_span,
                      bool& uturn_flag) {
  const Vec<S>& theta = last_span.theta_bk_;
  const Vec<S>& rho = last_span.rho_bk_;
  const Vec<S>& grad_theta = last_span.grad_theta_bk_;
  if (depth == 0) {
    Vec<S> theta_next;
    Vec<S> rho_next;
    Vec<S> grad_theta_next;
    S logp_theta_next;
    leapfrog(logp_grad_fun, inv_mass, -step, theta, rho, grad_theta,
             theta_next, rho_next, grad_theta_next, logp_theta_next);
    uturn_flag = false;
    return Span<S>(theta_next, rho_next, grad_theta_next, logp_theta_next);
  }
  Span<S> span1 = build_span_bk(rng, logp_grad_fun, inv_mass, step,
                                depth - 1, last_span, uturn_flag);
  if (uturn_flag) {
    return last_span;  // dummy
  }
  Span<S> span2 = build_span_bk(rng, logp_grad_fun, inv_mass, step,
                                depth - 1, span1, uturn_flag);
  if (uturn_flag) {
    return last_span; // dummy
  }
  if (uturn(span2.theta_bk_, span2.rho_bk_, span1.theta_fw_, span1.rho_fw_, inv_mass)) {
    uturn_flag = true;
    return last_span;  // dummy
  }
  return combine_bk<false>(rng, std::move(span2), std::move(span1), inv_mass, uturn_flag);
}

template <typename S, class RNG, class F>
Span<S> build_span_fw(RNG& rng,
                      const F& logp_grad_fun,
                      const Vec<S>& inv_mass,
                      S step,
                      int depth,
                      const Span<S>& last_span,
                      bool& uturn_flag) {
  const Vec<S>& theta = last_span.theta_fw_;
  const Vec<S>& rho = last_span.rho_fw_;
  const Vec<S>& grad_theta = last_span.grad_theta_fw_;
  if (depth == 0) {
    Vec<S> theta_next;
    Vec<S> rho_next;
    Vec<S> grad_theta_next;
    S logp_theta_next;
    leapfrog(logp_grad_fun, inv_mass, step, theta, rho, grad_theta,
             theta_next, rho_next, grad_theta_next, logp_theta_next);
    uturn_flag = false;
    return Span<S>(theta_next, rho_next, grad_theta_next, logp_theta_next);
  }
  Span<S> span1 = build_span_fw(rng, logp_grad_fun, inv_mass, step,
                                depth - 1, last_span, uturn_flag);
  if (uturn_flag) {
    return last_span;  // won't be used
  }
  Span<S> span2 = build_span_fw(rng, logp_grad_fun, inv_mass, step,
                                depth - 1, span1, uturn_flag);
  if (uturn_flag) {
    return last_span; // won't be used
  }
  if (uturn(span1.theta_bk_, span1.rho_bk_, span2.theta_fw_, span2.rho_fw_, inv_mass)) {
    uturn_flag = true;
    return last_span;  // won't be used
  }
  return combine_fw<false>(rng, std::move(span1), std::move(span2), inv_mass, uturn_flag);
}

template <typename S, typename RNG, class F>
void transition(RNG& rng,
                const F& logp_grad_fun,
                const Vec<S>& inv_mass,
                S step,
                int max_depth,
                const Vec<S>& theta,
                Vec<S>& theta_next) {
  static std::normal_distribution std_normal{0.0, 1.0};
  static std::uniform_int_distribution uniform_binary{0, 1};
  Vec<S> rho(theta.size());
  for (int i = 0; i < rho.size(); ++i) {
    rho(i) = std_normal(rng);
  }
  S logp;
  Vec<S> grad;
  logp_grad_fun(theta, logp, grad);
  logp -= kinetic(rho, inv_mass);
  Span<S> span_accum(theta, rho, grad, logp);

  for (int depth = 0; depth < max_depth; ++depth) {
    bool go_forward = uniform_binary(rng);
    bool uturn_flag;
    if (go_forward) {
      Span<S> span_next = build_span_fw(rng, logp_grad_fun, inv_mass, step, depth, span_accum, uturn_flag);
      if (uturn_flag) break;
      span_accum = combine_fw<true>(rng, std::move(span_accum), std::move(span_next), inv_mass, uturn_flag);
      if (uturn_flag) break;
    } else {
      Span<S> span_next = build_span_bk(rng, logp_grad_fun, inv_mass, step, depth, span_accum, uturn_flag);
      if (uturn_flag) break;
      span_accum = combine_bk<true>(rng, std::move(span_next), std::move(span_accum), inv_mass, uturn_flag);
      if (uturn_flag) break;
    }
  }
  theta_next = span_accum.theta_select_;
}

template <typename S, typename RNG, class F>
void nuts(RNG& rng,
          const F& logp_grad_fun,
          const Vec<S>& inv_mass,
          S step,
          int max_depth,
          const Vec<S>& theta,
          Matrix<S>& sample) {
  int num_draws = sample.rows();
  if (num_draws == 0) return;
  sample.row(0) = theta;
  Vec<S> theta_last = theta;
  Vec<S> theta_next;
  for (int n = 1; n < num_draws; ++n) {
    transition(rng, logp_grad_fun, inv_mass, step, max_depth, theta_last, theta_next);
    sample.row(n) = theta_next;
    theta_last = theta_next;
  }
}
