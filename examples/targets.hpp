#pragma once

using S = double;
using VectorS = Eigen::Matrix<S, -1, 1>;
using MatrixS = Eigen::Matrix<S, -1, -1>;
using Integer = long;

void std_normal_logp_grad(const VectorS& x, S& logp, VectorS& grad) {
  logp = -0.5 * x.dot(x);
  grad = -x;
}

void ill_cond_normal_logp_grad(const VectorS& x, S& logp, VectorS& grad) {
  Integer D = x.size();
  grad = VectorS::Zero(D);
  logp = 0;
  for (Integer d = 0; d < D; ++d) {
    double sigma = d + 1;
    double sigma_sq = sigma * sigma;
    logp += -0.5 * x[d] * x[d] / sigma_sq;
    grad[d] = -x[d] / sigma_sq;
  }
}

void corr_normal(const VectorS& x, S& logp, VectorS& grad) {
  S rho = 0.99;
  S tmp = 1 - std::square(rho);
  logp = -0.5 * std::square(x[0]) - 0.5 / tmp * std::square(x[1] - rho * x[0]);
  grad = VectorS::Zero(2);
  grad[0] = -(x[0] - rho * x[1]) / tmp;
  grad[1] = -(x[1] - rho * x[0]) / tmp;
}

void smile(const VectorS& x, S& logp, VectorS& grad) {
  lp = -0.5 * std::square(x[0]) - 0.5 * std::square(x[1] - std::square(x[0]));
  grad = VectorS::Zero(2);
  grad[0] = -x[0] + 2 * x[0] * (x[1] - std::square(x[0]));
  grad[1] = std::square(x[0]) - x[1];
}

S normal_logpdf(S x, S mu, S sigma) {
  return -0.5 * std::square((x - mu) / sigma) - std::log(sigma);
}

void funnel(const VectorS& x, S& logp, VectorS& grad) {
  logp = normal_logpdf(x[0], 0, 3);
  S sigma = std::exp(x[0] / 2);
  for (Integer n = 1; n < x.size(); ++n) {
    logp += normal_logpdf(x[n], 0, sigma);
  }
  grad = VectorS::Zero(x.size());
  S exp_neg_x0 = std::exp(-x[0]);
  S sum_sq = 0;
  for (n = 1; n < x.size(); ++n) {
    sum_sq += std::square(x[n]);
  }
  grad[0] = -(x.size() - 1) / 2.0 - x[0] / 9 + 0.5 * exp_neg_x0 * sum_sq;
  for (Integer n = 1; n < x.size(); ++n) {
    grad[n] = -x[n] * exp_neg_x0;
  }
}

def funnel10(q,hessian=False):
    lp=sps.norm.logpdf(q[0],loc=0.0,scale=3.0) + sum(sps.norm.logpdf(q[1:11],loc=0.0,scale=np.exp(0.5*q[0])))
    grad=np.r_[np.array([-5.0-q[0]/9.0 + 0.5*np.exp(-q[0])*sum(q[1:11]*q[1:11])])
               ,-q[1:11]*np.exp(-q[0])]
    return [lp,grad]
