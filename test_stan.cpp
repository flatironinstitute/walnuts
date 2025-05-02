#include <iostream>
#include <random>
#include <numeric>
#include <cmath>
#include <chrono>
#include "nuts.hpp"

#include "bridgestan.h"

double total_time = 0.0;
int count = 0;

#ifdef _WIN32
// hacky way to get dlopen and friends on Windows

#include <libloaderapi.h>
#include <errhandlingapi.h>
#define dlopen(lib, flags) LoadLibraryA(lib)
#define dlsym(handle, sym) (void*)GetProcAddress(handle, sym)

char* dlerror() {
  DWORD err = GetLastError();
  int length = snprintf(NULL, 0, "%d", err);
  char* str = malloc(length + 1);
  snprintf(str, length + 1, "%d", err);
  return str;
}
#else
#include <dlfcn.h>
#endif


class DynamicStanModel {
  public:
    DynamicStanModel(const char* model_path, const char* data, int seed) {
      handle_ = dlopen(model_path, RTLD_NOW);
      if (!handle_) {
        throw std::runtime_error("Error loading model: " + std::string(dlerror()));
      }

      model_construct = reinterpret_cast<decltype(&bs_model_construct)>(dlsym(handle_, "bs_model_construct"));
      free_error_msg = reinterpret_cast<decltype(&bs_free_error_msg)>(dlsym(handle_, "bs_free_error_msg"));
      model_destruct = reinterpret_cast<decltype(&bs_model_destruct)>(dlsym(handle_, "bs_model_destruct"));
      param_unc_num = reinterpret_cast<decltype(&bs_param_unc_num)>(dlsym(handle_, "bs_param_unc_num"));
      log_density_gradient = reinterpret_cast<decltype(&bs_log_density_gradient)>(dlsym(handle_, "bs_log_density_gradient"));


      if (!model_construct || !free_error_msg || !model_destruct || !param_unc_num || !log_density_gradient) {
        throw std::runtime_error("Error loading symbols: " + std::string(dlerror()));
      }

      char* err;
      model_ptr = model_construct(data, seed, &err);
      if (!model_ptr) {
        if (err) {
          std::string error_string(err);
          free_error_msg(err);
          throw std::runtime_error(error_string);
        }
        throw std::runtime_error("Failed to construct model");
      }
    }

    DynamicStanModel(const DynamicStanModel&) = delete; // non-copyable
    DynamicStanModel& operator=(const DynamicStanModel&) = delete; // non-copyable

    DynamicStanModel(DynamicStanModel&& other) :
      handle_(other.handle_),
      model_ptr(other.model_ptr),
      model_construct(other.model_construct),
      free_error_msg(other.free_error_msg),
      model_destruct(other.model_destruct),
      param_unc_num(other.param_unc_num),
      log_density_gradient(other.log_density_gradient)
      {}

    DynamicStanModel& operator=(DynamicStanModel&& other) {
      if (this != &other) {
        handle_ = other.handle_;
        model_ptr = other.model_ptr;
        model_construct = other.model_construct;
        free_error_msg = other.free_error_msg;
        model_destruct = other.model_destruct;
        param_unc_num = other.param_unc_num;
        log_density_gradient = other.log_density_gradient;

        other.handle_ = nullptr;
        other.model_ptr = nullptr;
      }
      return *this;
    }

    ~DynamicStanModel() {
      if (handle_) {
        if (model_ptr) {
          model_destruct(model_ptr);
        }
        dlclose(handle_);
      }
    }

    int size() const {
      return param_unc_num(model_ptr);
    }

    void logp_grad(const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
      double& logp,
      Eigen::Matrix<double, Eigen::Dynamic, 1>& grad) const {

      grad.resizeLike(x);

      char* err;
      auto start = std::chrono::high_resolution_clock::now();
      int ret = log_density_gradient(model_ptr, true, true, x.data(), &logp, grad.data(), &err);
      auto end = std::chrono::high_resolution_clock::now();
      total_time += std::chrono::duration<double>(end - start).count();
      ++count;

      if (ret != 0) {
        if (err) {
          std::string error_string(err);
          free_error_msg(err);
          throw std::runtime_error(error_string);
        }
        throw std::runtime_error("Failed to compute log density and gradient");
      }
    }

    private:
    void* handle_;
    bs_model* model_ptr;
    decltype(&bs_model_construct) model_construct;
    decltype(&bs_free_error_msg) free_error_msg;
    decltype(&bs_model_destruct) model_destruct;
    decltype(&bs_param_unc_num) param_unc_num;
    decltype(&bs_log_density_gradient) log_density_gradient;
};


int main(int argc, char* argv[]) {
  int init_seed = 333456;
  int seed = 763545;
  int N = 10000;
  double step_size = 0.025;
  int max_depth = 10;


  char* lib;
  char* data;

  // require at least the library name
  if (argc > 2) {
    lib = argv[1];
    data = argv[2];
  } else if (argc > 1) {
    lib = argv[1];
    data = NULL;
  } else {
    std::cerr << "Usage: " << argv[0] << " <model_path> [data]" << std::endl;
    return 1;
  }

  DynamicStanModel model(lib, data, init_seed);

  int D = model.size();

  std::cout << "D = " << D << ";  N = " << N
            << ";  step_size = " << step_size << ";  max_depth = " << max_depth
            << std::endl;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> draws(D, N);

  std::mt19937 rng(init_seed);
  std::normal_distribution<> std_normal(0.0, 1.0);
  Eigen::VectorXd theta_init(D);
  for (int i = 0; i < D; ++i) {
    theta_init(i) = std_normal(rng);
  }

  Eigen::VectorXd inv_mass = Eigen::VectorXd::Ones(D);

  auto global_start = std::chrono::high_resolution_clock::now();
  nuts(seed, [&model](const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
    double& logp,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& grad) { model.logp_grad(x,logp,grad);}, inv_mass, step_size, max_depth, theta_init, draws);
  auto global_end = std::chrono::high_resolution_clock::now();
  auto global_total_time = std::chrono::duration<double>(global_end - global_start).count();

  std::cout << "total time: " << global_total_time << "s" << std::endl;
  std::cout << "    gradient time: " << total_time << "s" << std::endl;
  std::cout << "        gradient calls: " << count << std::endl;
  std::cout << "        gradient time per call: " << total_time / count << "s" << std::endl;
  std::cout << std::endl;

  for (int d = 0; d < std::min(D, 10); ++d) {
    double mean = draws.row(d).mean();
    double var = (draws.row(d).array() - mean).square().sum() / (N - 1);
    double stddev = std::sqrt(var);
    std::cout << "dim " << d << ": mean = " << mean << ", stddev = " << stddev << "\n";
  }


  return 0;
}
