#include <cmath>
#include <random>

#include <Eigen/Dense>
#include <boost/ut.hpp>

#include <walnuts/dual_average.hpp>

namespace dual_average_test {

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

using namespace boost::ut;

suite<"dual_average"> tests = [] {
  "metropolis_1d"_test = [] {
    // theory says that if we target 0.44 accept rate, the step size will
    // be 2.4
    unsigned int seed = 7635445;
    std::mt19937 rng(seed);

    double delta = 0.44;  // optimal acceptance probability for D=1
    double init = 1.0;
    double t0 = 10.0;    // equal to default from Stan's NUTS
    double gamma = 0.1;  // equal to default from Stan's NUTS
    double kappa = 0.9;  // higher than default from Stan's NUTS
    nuts::DualAverage<double> da(init, delta, t0, gamma, kappa);
    int N = 100000;  // large N to account for different random behavior on
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
    expect(approx(2.4, step_size_hat, 0.2));  // step size not so accurate
    double accept_hat = total / count;
    expect(
        approx(delta, accept_hat, 0.01));  // achieved acceptance very accurate
  };
};
}  // namespace dual_average_test
