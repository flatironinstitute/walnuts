#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <walnuts.hpp>

double geom_mean_step(const std::vector<walnuts::ChainStore>& handlers) {
  if (handlers.size() == 0) {
    return 0.0;
  }
  double sum = 0;
  for (const auto& handler : handlers) {
    sum += std::log(handler.step_size());
  }
  return std::exp(sum / handlers.size());
}

Eigen::VectorXd geom_mean_inv_mass(
    const std::vector<walnuts::ChainStore>& handlers) {
  if (handlers.size() == 0) {
    return {};
  }
  Eigen::VectorXd sum(handlers[0].diag_inv_mass().rows());
  for (const auto& handler : handlers) {
    sum += handler.diag_inv_mass().array().log().matrix();
  }
  return (sum / handlers.size()).array().exp().matrix();
}

// 0) TARGET DENSITY ===========================================================
static void std_normal(const Eigen::VectorXd& x, double& lp,
                       Eigen::VectorXd& grad) {
  lp = -0.5 * x.dot(x);
  grad = -x;
}

int main() {
  // 1) CONFIGUFRE =============================================================
  auto logp_grad = std_normal;

  std::size_t seed = 48;
  std::seed_seq seed_seq_for_init{seed, static_cast<std::size_t>(0)};
  std::mt19937 rng{seed_seq_for_init};
  std::size_t num_chains = 32;
  std::size_t dims = 100;

  walnuts::CppInterruptCallback interrupt_callback;
  walnuts::GlobalStore global_handler;
  std::vector<walnuts::ChainStore> chain_handlers(num_chains);

  double init_scale = 0.5;
  double mass_smoothing = 0.1;
  auto init_cfg = walnuts::InitConfigBuilder(num_chains, dims)
                      .positions(rng, init_scale)
                      .masses(std_normal, mass_smoothing)
                      .build();

  auto warmup_cfg = walnuts::WarmupConfigBuilder()
                        .min_max_iter(50, 2000)
                        .mass_converge_tol(2.0)
                        .step_size_converge_tol(0.2)
                        .mass_init_count(4.0)
                        .build();

  auto sampling_cfg = walnuts::SamplingConfigBuilder()
                          .min_max_iter(50, 10000)
                          .max_trajectory_doublings(8)
                          .rhat_converge_tol(1.001)
                          .build();

  // std::cout << init_cfg << "\n\n";  // too verbose with multi-chain
  std::cout << warmup_cfg << "\n\n";
  std::cout << sampling_cfg << "\n\n";

  // 2) SAMPLE =================================================================
  // output sent to handlers
  walnuts::WalnutsConfig config{std::move(init_cfg), std::move(warmup_cfg),
                                std::move(sampling_cfg)};
  walnuts::walnuts<std::mt19937_64>(seed, chain_handlers, global_handler,
                                    interrupt_callback, logp_grad, config);

  // 3) SUMMARIZE ==============================================================
  std::cout << "ADAPTATION RESULT: " << "\n";
  std::cout << "  geom_mean(step_size) = " << geom_mean_step(chain_handlers)
            << "\n";
  std::cout << "  geom_mean(inv_mass) = "
            << geom_mean_inv_mass(chain_handlers).transpose() << "\n\n";

  std::cout << "PER-CHAIN STATISTICS: " << "\n";
  for (size_t m = 0; m < num_chains; ++m) {
    std::cout
        << "  Chain " << m << "; step size = " << chain_handlers[m].step_size()
        << "; ||mass|| = "
        << chain_handlers[m].diag_inv_mass().array().inverse().matrix().norm()
        << "; # warmup_draws = " << chain_handlers[m].warmup_draws().size()
        << "; # draws = " << chain_handlers[m].draws().size() << "\n";
  }
  std::cout << "\n";

  std::cout << "NUMBER OF R-HAT EVALS: " << global_handler.r_hats().size()
            << ";  FINAL R-HAT: " << global_handler.r_hats().back() << "\n\n";

  std::cout << "WRITING BINARY TO FILES: step_size.wal, mass_matrix.wal, "
               "sample.wal\n\n";

  walnuts::write_step_size("step_size.wal", chain_handlers);
  walnuts::write_mass_matrix("mass_matrix.wal", chain_handlers);
  walnuts::write_sample("sample.wal", chain_handlers);

  std::cout << "FINISHED NORMALLY." << std::endl << std::endl;
}
