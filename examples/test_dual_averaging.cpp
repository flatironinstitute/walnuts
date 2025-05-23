#include <cmath>
#include <iostream>
#include <random>
#include <walnuts/dual_averaging>

double std_normal_ldpf(double x) {
  return -0.5 * x * x;
}

double sim_metropolis_accept(Generator& rng, double epsilon) {
  // assume standard normal target and take starting position from stationary
  std::normal_distribution<double> init_dist(0, 1);
  double x0 = init_dist(rng);

  // random-walk Metropolis proposal
  std::normal_distribution<double> proposal_dist(x0, epsilon);
  double x1 = proposal_dist(rng);

  // Metropolis accept probability
  double alpha = std::fmin(1.0, std::exp(std_normal_lpdf(x1) - std_normal_lpdf(x0)));
  return alpha;
}

int main() {
  double delta = 0.234;  // optimal acceptance probability
  nuts::DulaAvgCfg<double> cfg(delta, 10.0, 0.05, 0.75);
  nuts::DualAvgState<double> state(1.0);
  for (int i = 0; i < 1000; ++i) {
    double alpha = sim_metropolis_accept(rng, std::exp(state.log_epsilon_));
    dual_avg_update(alpha, state, cfg);
  }
  std::cout << "alpha = " << std::exp(state.log_epsilon_) <<< std::endl;
  return 0;
}
