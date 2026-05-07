#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "walnuts/api.hpp"
#include "walnuts/config.hpp"
#include "walnuts/handlers.hpp"

// 0) TARGET DENSITY ===========================================================
static void std_normal(const Eigen::VectorXd& x, double& lp,
                       Eigen::VectorXd& grad) {
  lp = -0.5 * x.dot(x);
  grad = -x;
}

int main() {
  // 1) CONFIGUFRE =============================================================
  auto logp_grad = std_normal;

  uint32_t seed = 48;
  std::seed_seq seed_seq_for_init{seed, 0u};
  std::mt19937 rng{seed_seq_for_init};
  uint64_t num_chains = 32;
  uint64_t dims = 100;

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
                        .step_size_converge_tol(1)
                        .mass_init_count(4.0)
                        .build();

  auto sampling_cfg = walnuts::SamplingConfigBuilder()
                          .min_max_iter(50, 1000)
                          .max_trajectory_doublings(8)
                          .rhat_converge_tol(1.0001)
                          .build();

  // std::cout << init_cfg << "\n\n";  // too verbose with multi-chain
  std::cout << warmup_cfg << "\n\n";
  std::cout << sampling_cfg << "\n\n";

  // 2) SAMPLE =================================================================
  walnuts::AdaptResult result
    = walnuts::walnuts(seed, chain_handlers, global_handler, interrupt_callback,
		       logp_grad, init_cfg, warmup_cfg, sampling_cfg);

  // 3) SUMMARIZE ==============================================================
  std::cout << "ADAPTATION RESULT: " << "\n";
  std::cout << "  step_bar = " << result.step_bar << "\n";
  std::cout << "  mass_bar = " << result.mass_bar.transpose() << "\n\n";
  
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

  // CSV output is slowwwwwww
  // std::cout
  //     << "WRITING CSV TO FILES: step_size.csv, mass_matrix.csv,
  //     sample.csv\n\n";

  // walnuts::write_step_size_csv("step_size.csv", chain_handlers);
  // walnuts::write_mass_matrix_csv("mass_matrix.csv", chain_handlers);
  // walnuts::write_sample_csv("sample.csv", chain_handlers);

  std::cout << "WRITING BINARY TO FILES: step_size.wal, mass_matrix.wal, "
               "sample.wal\n\n";

  walnuts::write_step_size("step_size.wal", chain_handlers);
  walnuts::write_mass_matrix("mass_matrix.wal", chain_handlers);
  walnuts::write_sample("sample.wal", chain_handlers);

  std::cout << "FINISHED NORMALLY." << std::endl << std::endl;
}
