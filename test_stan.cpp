#include <iostream>
#include <random>
#include <cmath>
#include <chrono>

#include <Eigen/Dense>
#include <bridgestan.h>

#include "nuts.hpp"

// consider using something like https://github.com/martin-olivier/dylib/
#ifdef _WIN32
// hacky way to get dlopen and friends on Windows

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


template<typename T>
auto dlsym_cast(void* handle, T&&, const char* name) {
  auto sym = dlsym(handle, name);
  if (!sym) {
    throw std::runtime_error(std::string("Error loading symbol '") + name + "': " + dlerror());
  }
  return reinterpret_cast<T>(sym);
}

double total_time = 0.0;
int count = 0;

class DynamicStanModel {
public:
  DynamicStanModel(const char *model_path, const char *data, int seed) {
    library = dlopen(model_path, RTLD_NOW);
    if (!library) {
      throw std::runtime_error("Error loading model: " +
                               std::string(dlerror()));
    }

    model_construct =
        dlsym_cast(library, &bs_model_construct, "bs_model_construct");
    free_error_msg =
        dlsym_cast(library, &bs_free_error_msg, "bs_free_error_msg");
    model_destruct =
        dlsym_cast(library, &bs_model_destruct, "bs_model_destruct");
    param_unc_num =
        dlsym_cast(library, &bs_param_unc_num, "bs_param_unc_num");
    log_density_gradient =
        dlsym_cast(library, &bs_log_density_gradient, "bs_log_density_gradient");

    char *err = nullptr;
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

  // non-copyable
  DynamicStanModel(const DynamicStanModel &) = delete;
  DynamicStanModel &operator=(const DynamicStanModel &) = delete;

  DynamicStanModel(DynamicStanModel &&other)
      : library(other.library), model_ptr(other.model_ptr),
        model_construct(other.model_construct),
        free_error_msg(other.free_error_msg),
        model_destruct(other.model_destruct),
        param_unc_num(other.param_unc_num),
        log_density_gradient(other.log_density_gradient) {
    other.library = nullptr;
    other.model_ptr = nullptr;
  }

  DynamicStanModel &operator=(DynamicStanModel &&other) {
    if (this != &other) {
      if (library) {
        if (model_ptr) {
          model_destruct(model_ptr);
        }
        dlclose(library);
      }

      library = other.library;
      model_ptr = other.model_ptr;
      model_construct = other.model_construct;
      free_error_msg = other.free_error_msg;
      model_destruct = other.model_destruct;
      param_unc_num = other.param_unc_num;
      log_density_gradient = other.log_density_gradient;

      other.library = nullptr;
      other.model_ptr = nullptr;
    }
    return *this;
  }

  ~DynamicStanModel() {
    if (library) {
      if (model_ptr) {
        model_destruct(model_ptr);
      }
      dlclose(library);
    }
  }

  int size() const { return param_unc_num(model_ptr); }

  template <typename M>
  void logp_grad(const M &x, double &logp, M &grad) const {
    grad.resizeLike(x);

    char *err = nullptr;
    auto start = std::chrono::high_resolution_clock::now();
    int ret = log_density_gradient(model_ptr, true, true, x.data(), &logp,
                                   grad.data(), &err);
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
  void *library;
  bs_model *model_ptr;
  decltype(&bs_model_construct) model_construct;
  decltype(&bs_free_error_msg) free_error_msg;
  decltype(&bs_model_destruct) model_destruct;
  decltype(&bs_param_unc_num) param_unc_num;
  decltype(&bs_log_density_gradient) log_density_gradient;
};

int main(int argc, char *argv[]) {
  int seed = 333456;
  int N = 10000;
  double step_size = 0.025;
  int max_depth = 10;

  char *lib;
  char *data;

  if (argc <= 1) {
    // require at least the library name
    std::cerr << "Usage: " << argv[0] << " <model_path> [data]" << std::endl;
    return 1;
  } else if (argc == 2) {
    lib = argv[1];
    data = NULL;
  } else {
    lib = argv[1];
    data = argv[2];
  }

  DynamicStanModel model(lib, data, seed);

  int D = model.size();

  std::cout << "D = " << D << ";  N = " << N << ";  step_size = " << step_size
            << ";  max_depth = " << max_depth << std::endl;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> draws(D, N);

  Eigen::VectorXd inv_mass = Eigen::VectorXd::Ones(D);
  std::mt19937 generator(seed);
  std::normal_distribution<double> std_normal(0.0, 1.0);
  Eigen::VectorXd theta_init(D);
  for (int i = 0; i < D; ++i) {
    theta_init(i) = std_normal(generator);
  }

  auto global_start = std::chrono::high_resolution_clock::now();
  nuts::nuts(
      generator, [&model](auto &&...args) { model.logp_grad(args...); },
      inv_mass, step_size, max_depth, theta_init, draws);
  auto global_end = std::chrono::high_resolution_clock::now();
  auto global_total_time =
      std::chrono::duration<double>(global_end - global_start).count();

  std::cout << "    total time: " << global_total_time << "s" << std::endl;
  std::cout << "logp_grad time: " << total_time << "s" << std::endl;
  std::cout << "logp_grad fraction: " << total_time / global_total_time << std::endl;
  std::cout << "        logp_grad calls: " << count << std::endl;
  std::cout << "        time per call: " << total_time / count << "s" << std::endl;
  std::cout << std::endl;

  for (int d = 0; d < std::min(D, 10); ++d) {
    auto mean = draws.row(d).mean();
    auto var = (draws.row(d).array() - mean).square().sum() / (N - 1);
    auto stddev = std::sqrt(var);
    std::cout << "dim " << d << ": mean = " << mean << ", stddev = " << stddev
              << std::endl;
  }
  return 0;
}
