#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <walnuts/config.hpp>

namespace nuts {


  template <typename Adapter>
  void adapt(std::vector<Adapter>& adapters) {

  }
  
  template <typename Handler, typename RNG, typename LogProbGrad>
  void walnuts(RNG& rng,
	       std::vector<Handler>& handlers,  
	       const LogProbGrad& log_p_grad,
	       const InitConfig& init_config,
	       const WarmupConfig& warmup_config,
	       const SamplingConfig& sampling_config) {
    std::cout << init_config << std::endl;
    std::cout << warmup_config << std::endl;
    std::cout << sampling_config << std::endl;

  }

}  // namespace walnuts
