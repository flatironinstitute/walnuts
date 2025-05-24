#ifndef NUTS_UTIL_HPP
#define NUTS_UTIL_HPP

#include <Eigen/Dense>
#include <random>

namespace nuts {

template <typename S>
using Vec = Eigen::Matrix<S, Eigen::Dynamic, 1>;

template <typename S>
using Matrix = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;

using Integer = std::int32_t;

enum class Update { Barker, Metropolis };

enum class Direction { Backward, Forward };

template <typename S, class Generator>
class Random {
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

template <typename S>
S log_sum_exp(const S &x1, const S &x2) {
  using std::fmax, std::log, std::exp;
  S m = fmax(x1, x2);
  return m + log(exp(x1 - m) + exp(x2 - m));
}

template <typename S>
S log_sum_exp(const Vec<S> &x) {
  using std::log;
  S m = x.maxCoeff();
  return m + log((x.array() - m).exp().sum());
}

template <typename S>
S logp_momentum(const Vec<S> &rho, const Vec<S> &inv_mass) {
  return -0.5 * rho.dot(inv_mass.cwiseProduct(rho));
}

template <Direction D, typename S>
inline auto order_forward_backward(S &&s1, S &&s2) {
  if constexpr (D == Direction::Forward) {
    return std::forward_as_tuple(std::forward<S>(s1), std::forward<S>(s2));
  } else {  // Direction::Backward
    return std::forward_as_tuple(std::forward<S>(s2), std::forward<S>(s1));
  }
}

// U is either Span or WSpan; order_forward_backward generic
template <Direction D, typename S, class U>
inline bool uturn(const U &span_1, const U &span_2, const Vec<S> &inv_mass) {
  auto &&[span_bk, span_fw] = order_forward_backward<D>(span_1, span_2);
  auto scaled_diff =
      (inv_mass.array() * (span_fw.theta_fw_ - span_fw.theta_bk_).array())
          .matrix();
  return span_fw.rho_fw_.dot(scaled_diff) < 0 ||
         span_bk.rho_bk_.dot(scaled_diff) < 0;
}

}  // namespace nuts

#endif
