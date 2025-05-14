#include <walnuts/nuts.hpp>
#include <walnuts/walnuts.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>

using S = double;
using VectorS = Eigen::Matrix<S, -1, 1>;
using MatrixS = Eigen::Matrix<S, -1, -1>;

enum class Sampler { Nuts, Walnuts };


double total_time = 0.0;
int count = 0;

template <typename T>
void standard_normal_logp_grad(const Eigen::Matrix<T, Eigen::Dynamic, 1> &x,
                               S &logp,
                               Eigen::Matrix<T, Eigen::Dynamic, 1> &grad) {
  auto start = std::chrono::high_resolution_clock::now();
  logp = -0.5 * x.dot(x);
  grad = -x;
  auto end = std::chrono::high_resolution_clock::now();
  total_time += std::chrono::duration<double>(end - start).count();
  ++count;
}

template <Sampler U, typename G>
void test_nuts(const VectorS& theta_init, G& generator, int D, int N, S step_size, S max_depth,
	       S max_error, const VectorS& inv_mass) {
  total_time = 0.0;
  count = 0;
  MatrixS draws(D, N);
  std::cout << std::endl
	    << "D = " << D
	    << ";  N = " << N
	    << ";  step_size = " << step_size
            << ";  max_depth = " << max_depth
	    << ";  WALNUTS = " << (U == Sampler::Walnuts ? "true" : "false")
	    << std::endl;

  auto global_start = std::chrono::high_resolution_clock::now();
  if constexpr (U == Sampler::Walnuts) {
    walnuts::walnuts(generator, standard_normal_logp_grad<S>, inv_mass, step_size,
		     max_depth, max_error, theta_init, draws);
  } else if constexpr (U == Sampler::Nuts) {
    nuts::nuts(generator, standard_normal_logp_grad<S>, inv_mass, step_size,
	       max_depth, theta_init, draws);
  }
  auto global_end = std::chrono::high_resolution_clock::now();
  auto global_total_time =
      std::chrono::duration<double>(global_end - global_start).count();

  std::cout << "    total time: " << global_total_time << "s" << std::endl;
  std::cout << "logp_grad time: " << total_time << "s" << std::endl;
  std::cout << "logp_grad fraction: " << total_time / global_total_time
            << std::endl;
  std::cout << "        logp_grad calls: " << count << std::endl;
  std::cout << "        time per call: " << total_time / count << "s"
            << std::endl;
  std::cout << std::endl;

  for (int d = 0; d < std::min(D, 5); ++d) {
    auto mean = draws.row(d).mean();
    auto var = (draws.row(d).array() - mean).square().sum() / (N - 1);
    auto stddev = std::sqrt(var);
    std::cout << "dim " << d << ": mean = " << mean << ", stddev = " << stddev
              << "\n";
  }
  if (D > 10) {
    std::cout << "... elided " << (D - 5) << " dimensions ..." << std::endl;
  }
}

int main() {
  int seed = 428763;
  int D = 200;
  int N = 5000;
  S step_size = 0.25;
  int max_depth = 10;
  S max_error = 0.2;
  VectorS inv_mass = VectorS::Ones(D);

  std::mt19937 generator(seed);
  std::normal_distribution<S> std_normal(0.0, 1.0);
  VectorS theta_init(D);
  for (int i = 0; i < D; ++i) {
    theta_init(i) = std_normal(generator);
  }

  test_nuts<Sampler::Nuts>(theta_init, generator, D, N, step_size, max_depth, max_error, inv_mass);
  test_nuts<Sampler::Walnuts>(theta_init, generator, D, N, step_size, max_depth, max_error, inv_mass);
    
  return 0;
}
