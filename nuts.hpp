#include <cmath>
#include <random>
#include <utility>
#include <Eigen/Dense>

// void logp_grad(const vector& theta, vector& grad, double& lp)


template <typename Scalar>
Scalar log_sum_exp(const Scalar& x1, const Scalar& x2) {
  Scalar m = std::fmax(x1, x2);
  return m + std::log(exp(x1 - m) + exp(x2 - m));
}

template <typename Scalar>
Scalar log_sum_exp(const Eigen::Vector<Scalar>& x) {
  Scalar m = x.maxCoeff();
  return m + std::log((x - m).array().exp().vector().sum());
}

template <typename Scalar>
class span {
  const Eigen::Vector<Scalar> theta_bk_;
  const Eigen::Vector<Scalar> rho_bk_;
  const Eigen::Vector<Scalar> grad_theta_bk_;

  const Eigen::Vector<Scalar> theta_fw;
  const Eigen::Vector<Scalar> rho_fw_;
  const Eigen::Vector<Scalar> grad_theta_fw_;

  const Eigen::Vector<Scalar> theta_select_;
  Scalar logp_;

  span(const Eigen::Vector<Scalar>& theta_bk,
       const Eigen::Vector<Scalar>& rho_bk,
       const Eigen::Vector<Scalar>& grad_theta_bk,
       const Eigen::Vector<Scalar>& theta_fw,
       const Eigen::Vector<Scalar>& rho_fw,
       const Eigen::Vector<Scalar>& grad_theta_fw,
       const Eigen::Vector<Scalar>& theta_select,
       Scalar lp_total)
      : theta_bk_(theta_bk),
        rho_bk_(rho_bk),
        grad_theta_bk_(grad_theta_bk),
        theta_fw_(theta_fw),
        rho_fw_(rho_fw),
        grad_theta_fw_(grad_theta_fw),
        theta_select_(theta_select),
        logp_(lp_total)
  {}

  span(const Eigen::Vector<Scalar>& theta,
       const Eigen::Vector<Scalar>& rho,
       const Eigen::Vector<Scalar>& grad_theta,
       Scalar logp_theta)
      : span(theta, rho, grad_theta,
             theta, rho, grad_theta,
             theta, logp_theta)
  {}
};


template <typename Scalar, class Fun>
Scalar potential(const Eigen::Vector<Scalar>& theta,
                 const Fun& logp_grad) {
  Eigen::Vector<Scalar> grad;
  Scalar lp;
  logp_grad(theta, logp, grad);
  return -logp;
}


template <typename Scalar>
Scalar kinetic(const Eigen::Vector<Scalar>& rho,
               const Eigen::Vector<Scalar>& inv_mass) {
  return 0.5 * inv_mass.dot(rho.array().square().vector());
}


template <typename Scalar, class Fun>
Scalar hamiltonian(const Eigen::Vector<Scalar>& theta,
                   const Eigen::Vector<Scalar>& rho,
                   const Eigen::Vector<Scalar>& inv_mass,
                   const Fun& logp_grad_fun) {
  return potential(theta, logp) + kinetic(inv_mass, logp_grad_fun);
}


template <typename Scalar, typename Fun>
void leapfrog(Fun& logp_grad_fun,
              Eigen::Vector<Scalar>& inv_mass,
              Scalar step,
              const Eigen::Vector<Scalar>& theta,
              const Eigen::Vector<Scalar>& rho,
              const Eigen::Vector<Scalar>& grad_theta,
              Eigen::Vector<Scalar>& theta_next,
              Eigen::Vector<Scalar>& rho_next,
              Eigen::Vector<Scalar>& grad_theta_next
              Scalar& logp_theta_next) {
  Scalar half_step = 0.5 * step;
  auto step_inv_mass = step * inv_mass;
  rho_next = rho + half_step * grad_theta;
  theta_next = theta + step_inv_mass * rho;
  lp_grad_theta_next = lp_grad_fun(theta_next);
  rho_next += half_step * lp_grad_theta_next.grad;
}


template <typename Scalar>
bool uturn(const Eigen::Vector<Scalar>& theta_bk,
           const Eigen::Vector<Scalar>& rho_bk,
           const Eigen::Vector<Scalar>& theta_fw,
           const Eigen::Vector<Scalar>& rho_fw,
           const Eigen::Vector<Scalar>& inv_mass) {
  auto scaled_diff = (inv_mass.array() * (theta_bk - theta_fw).array()).vector();
  return rho_fw.dot(scaled_diff) < 0 || rho_bk.dot(scaled_diff) < 0;
}

template <bool Progressive, typename Scalar, typename RNG>
span combine(RNG& rng, const span& span1, const span& span2,
             Eigen::Vector<Scalar> inv_mass, bool& uturn) {
  if (uturn(span1.theta_bk, span1.rho_bk, span2.theta_fw, span2._rho_fw, inv_mass)) {
    uturn = true;
    return span1;
  }
  Scalar logp12 = log_sum_exp(span1.logp, span2.logp);
  Scalar log_denominator = Progressive ? span1.logp : logp12;
  Scalar update_prob = std::exp(span2.logp - log_denominator);
  std::uniform_real_distribution<> u(0.0, 1.0);
  bool update = u(rng) < update_prob;
  auto selected = update ? span2.theta_select : span1:theta_select;
  uturn = false;
  return span(span1.theta_bk, span1.rho_bk, span1.grad_theta_bk,
              span2.theta_fw, span2.rho_fw, span2.grad_theta_fw, logp12);
}

template <typename Scalar, class RNG, class Fun>
span build_span_bk(RNG& rng,
                   const Fun& logp_grad_fun,
                   const Eigen::Vector<Scalar>& inv_mass,
                   Scalar step,
                   int depth,
                   const span& last_span,
                   bool& uturn) {
  auto theta = last_span.theta_bk_;
  auto rho = last_span.rho_bk_;
  if (depth == 0) {
    Eigen::Vector<Scalar> theta_next;
    Eigen::Vector<Scalar> rho_next;
    Eigen::Vector<Scalar> grad_theta_next;
    Eigen::Vector<Scalar> logp_theta_next;
    leapfrog(logp_grad_fun, inv_mass, -step, theta, rho, grad_theta,
             theta_next, rho_next, grad_theta_next, logp_theta_next);
    uturn = false;
    return span(theta_next, rho_next, grad_theta_next, logp_theta_next);
  }
  span span1 = build_span_bk(rng, logp_grad_fun, inv_mass, step,
                             depth - 1, last_span, uturn);
  if (uturn) {
    return last_span;  // won't be used
  }
  span span2 = build_span_bk(rng, logp_grad_fun, inv_mass, step,
                             depth - 1, span1, uturn);
  if (uturn) {
    return last_span; // won't be used
  }
  if (uturn(span2.theta_bk_, span2.rho_bk_, span1.theta_fw_, span1.rho_fw_, inv_mass)) {
    uturn = true;
    return last_span;  // won't be used
  }
  return combine<false>(rng, span2, span1, inv_mass, uturn);
}

template <typename Scalar, class RNG, class Fun>
span build_span_fw(RNG& rng,
                   const Fun& logp_grad_fun,
                   const Eigen::Vector<Scalar>& inv_mass,
                   Scalar step,
                   int depth,
                   const span& last_span,
                   bool& uturn) {
  auto theta = last_span.theta_fw_;
  auto rho = last_span.rho_fw_;
  if (depth == 0) {
    Eigen::Vector<Scalar> theta_next;
    Eigen::Vector<Scalar> rho_next;
    Eigen::Vector<Scalar> grad_theta_next;
    Eigen::Vector<Scalar> logp_theta_next;
    leapfrog(logp_grad_fun, inv_mass, step, theta, rho, grad_theta,
             theta_next, rho_next, grad_theta_next, logp_theta_next);
    uturn = false
    return span(theta_next, rho_next, grad_theta_next, logp_theta_next);
  }
  span span1 = build_span_fw(rng, logp_grad_fun, inv_mass, step,
                             depth - 1, last_span, uturn);
  if (uturn) {
    return last_span;  // won't be used
  }
  span span2 = build_span_fw(rng, logp_grad_fun, inv_mass, step,
                             depth - 1, span1, uturn);
  if (uturn) {
    return last_span; // won't be used
  }
  if (uturn(span1.theta_bk_, span1.rho_bk_, span2.theta_fw_, span2.rho_fw_, inv_mass)) {
    uturn = true;
    return last_span;  // won't be used
  }
  return combine<false>(rng, span1, span2, inv_mass, uturn);
}



template <typename Scalar>
void transition(RNG& rng,
                const Fun& logp_grad_fun,
                const Eigen::Vector<Scalar>& inv_mass,
                Scalar step,
                int max_depth,
                const Eigen::Vector<Scalar>& theta,
                Eigen::Vector<Scalar>& theta_next) {
  Eigen::Vector<Scalar> rho(theta.size());
  std::normal_distribution std_normal{0.0, 1.0};
  std::uniform_int_distribution uniform_binary{0, 1};
  for (int i = 0; i < rho.size(); ++i) {
    rho(i) = std_normal(rng);
  }
  Scalar logp;
  const Eigen::Vector<Scalar> grad;
  logp_grad_fun(theta, rho, logp, grad);
  span span_accum(theta, rho, grad, logp);
  for (int depth = 0; depth < max_depth; ++depth) {
    int go_forward = uniform_binary(rng);
    bool uturn;
    if (go_forward) {
      span span_next = build_span_fw(rng, logp_grad_fun, inv_mass, step, depth, span_accum, uturn);
      if (uturn) break;
      span_accum = combine<true>(rng, span_accum, span_next, true, uturn);
      if (uturn) break;
    } else {
      span span_next = build_span_bk(rng, logp_grad_fun, inv_mass, step, depth, span_accum, uturn);
      if (uturn) break;
      span_accum = combine<true>(rng, span_next, span_accum, true, uturn);
      if (uturn) break;

    }
  }
  theta_next = span_accum.theta_selected_;
}

template <typename Scalar>
void nuts(RNG& rng,
          const Fun& logp_grad_fun,
          const Eigen::Vector<Scalar>& inv_mass,
          Scalar step,
          int max_depth,
          const Eigen::Vector<Scalar>& theta,
          Eigen::Matrix<Scalar, -1, -1>& sample) {
  int num_draws = sample.rows();
  if (num_draws == 0) return;
  sample.row(0) = theta;
  Eigen::Vector<Scalar> theta_last = theta;
  Eigen::Vector<Scalar> theta_next;
  for (int n = 1; n < num_draws; ++n) {
    transition(rng, logp_grad_fun, inv_mass, step, theta_last, theta_next);
    sample.row(n) = theta_next;
    theta_last = theta_next;
  }
}*

// NEED TO MAKE ARGS TO LEAPFROG, TRANSITION, ETC. MORE GENERIC
