#ifndef WALNUTS_RANDOM_HPP
#define WALNUTS_RANDOM_HPP

#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <cmath>
#include <random>

namespace nuts {

template <typename S> using Vec = Eigen::Matrix<S, Eigen::Dynamic, 1>;

template <typename S>
using Matrix = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;

using Integer = std::int32_t;

template <typename S, class Generator> class Random {
public:
  explicit Random(Generator &rng)
      : rng_(rng), unif_(0.0, 1.0), binary_(0.5), normal_() {}

  inline S uniform_real_01() { return unif_(rng_); }

  inline bool uniform_binary() { return binary_(rng_); }

  inline Vec<S> standard_normal(Integer n) {
    return normal_.template generate<Vec<S>>(n, 1, rng_);
  }

private:
  Generator &rng_;
  std::uniform_real_distribution<S> unif_;
  std::bernoulli_distribution binary_;
  Eigen::Rand::StdNormalGen<S> normal_;
};

enum class Update { Barker, Metropolis };

enum class Direction { Backward, Forward };

} // namespace nuts

#endif
