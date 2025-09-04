#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <walnuts/dual_average.hpp>

static double std_normal_lpdf(double x) { return -0.5 * x * x; }

template <class G>
static double sim_metropolis_accept(G& rng, double step_size) {
  // draw previous state from std normal target
  std::normal_distribution<double> init_dist(0, 1);
  double x0 = init_dist(rng);

  // random-walk Metropolis proposal
  std::normal_distribution<double> proposal_dist(x0, step_size);
  double x1 = proposal_dist(rng);

  // Metropolis accept probability
  return std::fmin(1.0, std::exp(std_normal_lpdf(x1) - std_normal_lpdf(x0)));
}

TEST(DualAverage, Metropolis1D) {
  // theory says that if we target 0.44 accept rate, the step size will be 2.4
  unsigned int seed = 7635445;
  std::mt19937 rng(seed);

  double delta = 0.44;  // optimal acceptance probability for D=1
  double init = 1.0;
  double t0 = 10.0;    // equal to default from Stan's NUTS
  double gamma = 0.1;  // equal to default from Stan's NUTS
  double kappa = 0.9;  // higher than default from Stan's NUTS
  nuts::DualAverage<double> da(init, delta, t0, gamma, kappa);
  int N = 1000000;  // large N to account for different random behavior on
                    // different OSes
  double total = 0;
  double count = 0;
  for (int n = 0; n < N; ++n) {
    double step_size_hat = da.step_size();
    double alpha = sim_metropolis_accept(rng, step_size_hat);
    da.observe(alpha);
    total += alpha;
    count += 1.0;
  }
  double step_size_hat = da.step_size();
  EXPECT_NEAR(2.4, step_size_hat, 0.2);  // step size not so accurate
  double accept_hat = total / count;
  EXPECT_NEAR(delta, accept_hat, 0.01);  // achieved acceptance very accurate
}

TEST(DualAverage, observe1) {
  unsigned int seed = 7635445;
  std::mt19937 rng(seed);

  double epsilon0 = 1.0;
  double delta = 0.8;
  double t0 = 10.0;
  double gamma = 0.05;
  double kappa = 1.5;
  nuts::DualAverage<double> da(epsilon0, delta, t0, gamma, kappa);

  EXPECT_NEAR(epsilon0, da.step_size(), 1e-10);

  double alpha = 0.2;
  for (int i = 0; i < 100; ++i)
    da.observe(alpha);
  // worked out answer by hand
  EXPECT_NEAR(3.359109812391624, da.step_size(), 1e-5);
}
