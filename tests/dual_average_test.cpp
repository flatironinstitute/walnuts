#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <walnuts/dual_average.hpp>

double std_normal_lpdf(double x) {
  return -0.5 * x * x;
}

template <class G>
double sim_metropolis_accept(G& rng, double epsilon) {
  // draw previous state from std normal target
  std::normal_distribution<double> init_dist(0, 1);
  double x0 = init_dist(rng);

  // random-walk Metropolis proposal
  std::normal_distribution<double> proposal_dist(x0, epsilon);
  double x1 = proposal_dist(rng);

  // Metropolis accept probability
  return std::fmin(1.0, std::exp(std_normal_lpdf(x1) - std_normal_lpdf(x0)));
}


TEST(DualAverage, Metropolis1D) {
  // theory says that if we target 0.44 accept rate, the step size will be 2.4
  unsigned int seed = 763545;
  std::mt19937 rng(seed);
  
  double delta = 0.44;  // optimal acceptance probability for D=1
  double init = 1.0;
  double t0 = 10.0;  // equal to default from Stan's NUTS
  double gamma = 0.1;  // equal to default from Stan's NUTS
  double kappa = 0.9;  // higher than default from Stan's NUTS
  nuts::DualAverage<double> da(init, delta, t0, gamma, kappa);
  int N = 1000;  // N = 100000 not much more accurate at these settings
  double total = 0;
  double count = 0;
  for (int n = 0; n < N; ++n) {
    double epsilon_hat = da.epsilon();
    double alpha = sim_metropolis_accept(rng, epsilon_hat);
    da.update(alpha);
    total += alpha;
    count += 1.0;
  }
  double epsilon_hat = da.epsilon();
  EXPECT_NEAR(2.4, epsilon_hat, 0.1);   // step size not so accurate
  double accept_hat = total / count;
  EXPECT_NEAR(0.44, accept_hat, 0.01);  // achieved acceptance very accurate
}
