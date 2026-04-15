#pragma once

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <latch>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <stop_token>
#include <thread>
#include <vector>
#include <Eigen/Dense>
#include <walnuts/config.hpp>

#include <walnuts/adapt.hpp>
#include <walnuts/config.hpp>
#include <walnuts/padded.hpp>
#include <walnuts/triple_buffer.hpp>

namespace walnuts {

  class TemporaryStubAdapter {
  public:
    TemporaryStubAdapter(std::size_t dim, std::uint64_t seed)
      : dim_(dim),
        rng_(seed),
        z_(0.0f, 1.0f),
        log_mass_means_(means(dim)),
        log_mass_(dim),
        log_step_(std::log(0.1f)) {}

    std::size_t dim() const noexcept { return dim_; }

    void step(std::uint32_t iter) {
      const float sd = 1.0f / std::sqrt(static_cast<float>(iter));
      log_step_ = std::log(0.1f) + sd * z_(rng_);
      for (std::size_t d = 0; d < dim_; ++d) {
	log_mass_[d] = log_mass_means_[d] + sd * z_(rng_);
      }
      ++iter_;
    }

    float log_step() const noexcept { return log_step_; }

    std::span<const float> log_mass() const noexcept { return log_mass_; }

    std::size_t iter() const noexcept { return iter_; }
  
  private:
    static std::vector<float> means(std::size_t dim) {
      std::vector<float> m(dim);
      for (std::size_t d = 0; d < dim; ++d) {
	const float x = static_cast<float>(d + 1);
	m[d] = std::log(x * x);
      }
      return m;
    }

    std::size_t iter_ = 0;
    std::size_t dim_;
    std::mt19937_64 rng_;
    std::normal_distribution<float> z_;
    std::vector<float> log_mass_means_;
    std::vector<float> log_mass_;
    float log_step_;
  };
  

  template <typename RNG, typename Handler, typename LogProbGrad>
  void walnuts(RNG& rng,
	       std::vector<Handler>& handlers,  
	       const LogProbGrad& log_p_grad,
	       const InitConfig& init_cfg,
	       const WarmupConfig& warmup_cfg,
	       const SamplingConfig& sampling_cfg) {
    std::cout << init_cfg << std::endl;
    std::cout << warmup_cfg << std::endl;
    std::cout << sampling_cfg << std::endl;

    std::vector<TemporaryStubAdapter> adapters;
    adapters.reserve(init_cfg.num_chains());
    for (std::size_t m = 0; m < init_cfg.num_chains(); ++m) {
      adapters.emplace_back(TemporaryStubAdapter(init_cfg.dims(), rng()));
    }

    AdaptResult res = adapt<TemporaryStubAdapter>(init_cfg, warmup_cfg, adapters);

    const float mass_bar_norm = l2_norm(std::span<const float>(res.mass_bar));
  
    std::cout << "\nSHARED ADAPTED RESULT:  "
	      << "stop_iter_min=" << res.stop_iter_min
	      << "  step_bar=" << res.step_bar
	      << "  ||mass_bar||=" << mass_bar_norm
	      << '\n';

    std::cout << "\nPER CHAIN FINAL STATES:\n";
    for (std::size_t m = 0; m < adapters.size(); ++m) {
      std::cout << m << ")"
		<< " iter = " << adapters[m].iter()
		<< "  step = " << std::exp(adapters[m].log_step())
		<< "  ||log_mass|| = " << l2_norm(std::span<const float>(adapters[m].log_mass()))
		<< std::endl;
    }
  }
  
}  // namespace walnuts
