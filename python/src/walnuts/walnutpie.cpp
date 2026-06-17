#include <cstddef>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <walnuts.hpp>

#include "errors.hpp"
#include "export.h"
#include "handlers.hpp"
#include "interrupts.hpp"

// TODO(bmw): find a better folder organization
#include "../examples/load_stan.hpp"
#include "walnuts/concepts.hpp"

namespace walnutpie {

void walnutpie_helper(const walnuts::LogpGrad auto& logp, int num_params,
                      walnuts::InitConfigBuilder& init_cfg_builder,
                      auto& handlers, size_t num_chains, unsigned int seed,
                      unsigned int id, const double* init_inv_metric,
                      int min_warmup_iter, int max_warmup_iter,
                      int min_sampling_iter, int max_sampling_iter,
                      int max_trajectory_doublings, int max_step_halvings,
                      int min_micro_steps, double max_hamiltonian_error,
                      double step_size_converge_tol, double mass_converge_tol,
                      double rhat_converge_tol, double mass_init_count,
                      double mass_additive_smoothing,
                      double max_macro_steps_target,
                      double step_accept_rate_target, double step_learning_rate,
                      double step_gradient_decay, double step_sq_gradient_decay,
                      double step_stabilization, double step_learn_rate_decay) {
  interrupt::walnutpie_interrupt_handler interrupt;

  walnuts::WarmupConfig warmup_cfg =
      walnuts::WarmupConfigBuilder()
          .min_max_iter(min_warmup_iter, max_warmup_iter)
          .step_size_converge_tol(step_size_converge_tol)
          .mass_converge_tol(mass_converge_tol)
          // .publish_stride(publish_stride)
          // .yield_period(yield_period)
          .mass_init_count(mass_init_count)
          .mass_additive_smoothing(mass_additive_smoothing)
          .max_macro_steps_target(max_macro_steps_target)
          .step_accept_rate_target(step_accept_rate_target)
          .step_learning_rate(step_learning_rate)
          .step_gradient_decay(step_gradient_decay)
          .step_sq_gradient_decay(step_sq_gradient_decay)
          .step_stabilization(step_stabilization)
          .step_learn_rate_decay(step_learn_rate_decay)
          .build();

  walnuts::SamplingConfig sample_cfg =
      walnuts::SamplingConfigBuilder()
          .min_max_iter(min_sampling_iter, max_sampling_iter)
          .rhat_converge_tol(rhat_converge_tol)
          .max_trajectory_doublings(max_trajectory_doublings)
          .max_step_halvings(max_step_halvings)
          .max_hamiltonian_error(max_hamiltonian_error)
          .min_micro_steps(min_micro_steps)
          .build();

  if (init_inv_metric != nullptr) {
    std::vector<Eigen::VectorXd> mass_inits(num_chains);
    for (size_t i = 0; i < num_chains; ++i) {
      mass_inits[i] = Eigen::Map<const Eigen::VectorXd>(
          init_inv_metric + (i * num_params), num_params);
    }
    init_cfg_builder.masses(mass_inits);
  } else {
    init_cfg_builder.masses(logp, mass_additive_smoothing);
  }

  walnuts::WalnutsConfig walnuts_cfg{
      init_cfg_builder.build(), std::move(warmup_cfg), std::move(sample_cfg)};

  DummyGlobalHandler global;
  walnuts::walnuts<std::mt19937_64>(seed + id + num_chains, handlers, global,
                                    interrupt, logp, walnuts_cfg);
}
}  // namespace walnutpie

using namespace walnutpie;

extern "C" {

/** Function pointer to a log density function
 * This is intentionally a mirror of `RawLogpFunc` in
 * nutpie (https://github.com/pymc-devs/nutpie/blob/main/src/pymc.rs#L23)
 */
typedef int (*LOGP_CFUNC)(size_t theta_size, const double* theta, double* grad,
                          double* lp, void* data);

WALNUTPIE_EXPORT int walnutpie_sample_cfunc(
    LOGP_CFUNC logp_c, void* data, int num_params, const double* inits,
    size_t num_chains, unsigned int seed, unsigned int id, double init_radius,
    const double* init_inv_metric, int min_warmup_iter, int max_warmup_iter,
    int min_sampling_iter, int max_sampling_iter, int max_trajectory_doublings,
    int max_step_halvings, int min_micro_steps, double max_hamiltonian_error,
    double step_size_converge_tol, double mass_converge_tol,
    double rhat_converge_tol, double mass_init_count,
    double mass_additive_smoothing, double max_macro_steps_target,
    double step_size_init, double step_accept_rate_target,
    double step_learning_rate, double step_gradient_decay,
    double step_sq_gradient_decay, double step_stabilization,
    double step_learn_rate_decay, bool save_warmup, int refresh, double* out,
    size_t out_size, int* final_lengths, double* stepsize_out,
    double* inv_metric_out, WalnutpieError** err) {
  return error::catch_exceptions(err, [&]() {
    int draws_offset =
        num_params * (max_sampling_iter + max_warmup_iter * save_warmup);
    if (out_size < num_chains * draws_offset) {
      std::stringstream ss;
      ss << "Output buffer too small. Expected at least " << num_chains
         << " chains of " << draws_offset << " doubles, got " << out_size;
      throw std::runtime_error(ss.str());
    }

    auto logp = [&](const Eigen::VectorXd& x, double& logp,
                    Eigen::VectorXd& grad) {
      grad.resizeLike(x);
      int ret = logp_c(x.size(), x.data(), grad.data(), &logp, data);
      if (ret != 0) {
        throw std::runtime_error("logp failed with code " +
                                 std::to_string(ret));
      }
    };

    auto init_cfg_builder =
        walnuts::InitConfigBuilder{num_chains,
                                   static_cast<std::size_t>(num_params)}
            .step_sizes(step_size_init);
    if (inits != nullptr) {
      std::vector<Eigen::VectorXd> theta_inits(num_chains);

      for (size_t i = 0; i < num_chains; ++i) {
        if (inits != nullptr) {
          theta_inits[i] = Eigen::Map<const Eigen::VectorXd>(
              inits + i * num_params, num_params);
        }
      }
      init_cfg_builder.positions(theta_inits);
    } else {
      std::seed_seq ss{seed, 1u};
      std::mt19937_64 rng(ss);
      init_cfg_builder.positions(rng, init_radius);
    }

    std::vector<BufferHandler> handlers;
    handlers.reserve(num_chains);
    for (size_t i = 0; i < num_chains; ++i) {
      handlers.emplace_back(
          out + draws_offset * i, num_params,
          stepsize_out != nullptr ? stepsize_out + i : nullptr,
          inv_metric_out != nullptr ? inv_metric_out + i * num_params : nullptr,
          save_warmup);
    }

    walnutpie_helper(
        logp, num_params, init_cfg_builder, handlers, num_chains, seed, id,
        init_inv_metric, min_warmup_iter, max_warmup_iter, min_sampling_iter,
        max_sampling_iter, max_trajectory_doublings, max_step_halvings,
        min_micro_steps, max_hamiltonian_error, step_size_converge_tol,
        mass_converge_tol, rhat_converge_tol, mass_init_count,
        mass_additive_smoothing, max_macro_steps_target,
        step_accept_rate_target, step_learning_rate, step_gradient_decay,
        step_sq_gradient_decay, step_stabilization, step_learn_rate_decay);

    for (size_t i = 0; i < num_chains; ++i) {
      final_lengths[i] = handlers[i].written();
    }

    return 0;
  });
}

WALNUTPIE_EXPORT int walnutpie_sample_bridgestan(
    const char* bs_dll, const char* json_data, unsigned int model_seed,
    const char* inits, size_t num_chains, unsigned int seed, unsigned int id,
    double init_radius, const double* init_inv_metric, int min_warmup_iter,
    int max_warmup_iter, int min_sampling_iter, int max_sampling_iter,
    int max_trajectory_doublings, int max_step_halvings, int min_micro_steps,
    double max_hamiltonian_error, double step_size_converge_tol,
    double mass_converge_tol, double rhat_converge_tol, double mass_init_count,
    double mass_additive_smoothing, double max_macro_steps_target,
    double step_size_init, double step_accept_rate_target,
    double step_learning_rate, double step_gradient_decay,
    double step_sq_gradient_decay, double step_stabilization,
    double step_learn_rate_decay, bool save_warmup, int refresh, double* out,
    size_t out_size, int* final_lengths, double* stepsize_out,
    double* inv_metric_out, WalnutpieError** err) {
  return error::catch_exceptions(err, [&]() {
    DynamicStanModel stan_model(bs_dll, json_data, model_seed);

    int draws_offset = stan_model.constrained_dimensions() *
                       (max_sampling_iter + max_warmup_iter * save_warmup);
    if (out_size < num_chains * draws_offset) {
      std::stringstream ss;
      ss << "Output buffer too small. Expected at least " << num_chains
         << " chains of " << draws_offset << " doubles, got " << out_size;
      throw std::runtime_error(ss.str());
    }

    auto logp = [&](auto&&... args) { stan_model.logp_grad(args...); };

    std::vector<unique_bs_rng> rngs;
    std::vector<Eigen::VectorXd> theta_inits;
    std::vector<StanBufferHandler> handlers;
    rngs.reserve(num_chains);
    theta_inits.reserve(num_chains);
    handlers.reserve(num_chains);
    {
      std::seed_seq ss{seed, 1u};
      std::vector<std::uint32_t> seeds(num_chains);
      ss.generate(seeds.begin(), seeds.end());
      for (size_t i = 0; i < num_chains; ++i) {
        rngs.push_back(stan_model.make_rng(seeds[i]));

        // TODO need some sort of init separation per chain. In tinystan used
        // a 'magic' separator
        theta_inits.push_back(
            stan_model.initialize(inits, rngs[i], init_radius));

        handlers.emplace_back(
            stan_model, rngs[i], out + draws_offset * i,
            stepsize_out != nullptr ? stepsize_out + i : nullptr,
            inv_metric_out != nullptr
                ? inv_metric_out + i * stan_model.unconstrained_dimensions()
                : nullptr,
            save_warmup);
      }
    }

    auto init_cfg_builder =
        walnuts::InitConfigBuilder{
            num_chains, static_cast<std::size_t>(theta_inits[0].size())}
            .step_sizes(step_size_init)
            .positions(theta_inits);

    walnutpie_helper(
        logp, stan_model.unconstrained_dimensions(), init_cfg_builder, handlers,
        num_chains, seed, id, init_inv_metric, min_warmup_iter, max_warmup_iter,
        min_sampling_iter, max_sampling_iter, max_trajectory_doublings,
        max_step_halvings, min_micro_steps, max_hamiltonian_error,
        step_size_converge_tol, mass_converge_tol, rhat_converge_tol,
        mass_init_count, mass_additive_smoothing, max_macro_steps_target,
        step_accept_rate_target, step_learning_rate, step_gradient_decay,
        step_sq_gradient_decay, step_stabilization, step_learn_rate_decay);

    for (size_t i = 0; i < num_chains; ++i) {
      final_lengths[i] = handlers[i].written();
    }

    return 0;
  });
}

WALNUTPIE_EXPORT const char* walnutpie_get_error_message(
    const WalnutpieError* err) {
  if (err == nullptr) {
    return "Something went wrong: No error found";
  }
  return err->msg.c_str();
}

WALNUTPIE_EXPORT WalnutpieErrorType
walnutpie_get_error_type(const WalnutpieError* err) {
  if (err == nullptr) {
    return WalnutpieErrorType::generic;
  }
  return err->type;
}

WALNUTPIE_EXPORT void walnutpie_destroy_error(WalnutpieError* err) {
  delete (err);
}
}
