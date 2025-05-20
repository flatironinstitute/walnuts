#include <bridgestan.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <walnuts/nuts.hpp>
#include <walnuts/walnuts.hpp>

// consider using something like https://github.com/martin-olivier/dylib/
#ifdef _WIN32
// hacky way to get dlopen and friends on Windows

#include <errhandlingapi.h>
#define dlopen(lib, flags) LoadLibraryA(lib)
#define dlsym(handle, sym) (void *)GetProcAddress(handle, sym)
#define dlclose(handle) FreeLibrary(handle)

char *dlerror() {
  DWORD err = GetLastError();
  int length = snprintf(NULL, 0, "%d", err);
  char *str = malloc(length + 1);
  snprintf(str, length + 1, "%d", err);
  return str;
}
#else
#include <dlfcn.h>
#endif

struct dlclose_deleter {
  void operator()(void *handle) const {
    if (handle) {
      dlclose(handle);
    }
  }
};

auto dlopen_safe(const char *path) {
  auto handle = dlopen(path, RTLD_NOW);
  if (!handle) {
    throw std::runtime_error(std::string("Error loading library '") + path +
                             "': " + dlerror());
  }
  return std::unique_ptr<void, dlclose_deleter>(handle);
}

template <typename U, typename T>
auto dlsym_cast(U &library, T &&, const char *name) {
  auto sym = dlsym(library.get(), name);
  if (!sym) {
    throw std::runtime_error(std::string("Error loading symbol '") + name +
                             "': " + dlerror());
  }
  return reinterpret_cast<T>(sym);
}

double total_time = 0.0;
int count = 0;

template <typename T>
void no_op_deleter(T *) {}

class DynamicStanModel {
 public:
  DynamicStanModel(const char *model_path, const char *data, int seed)
      : library_(dlopen_safe(model_path)),
        model_ptr_(nullptr, no_op_deleter<bs_model>),
        rng_ptr_(nullptr, no_op_deleter<bs_rng>) {
    auto model_construct =
        dlsym_cast(library_, &bs_model_construct, "bs_model_construct");
    auto model_destruct =
        dlsym_cast(library_, &bs_model_destruct, "bs_model_destruct");
    auto rng_construct =
        dlsym_cast(library_, &bs_rng_construct, "bs_rng_construct");
    auto rng_destruct =
        dlsym_cast(library_, &bs_rng_destruct, "bs_rng_destruct");

    free_error_msg_ =
        dlsym_cast(library_, &bs_free_error_msg, "bs_free_error_msg");
    param_unc_num_ =
        dlsym_cast(library_, &bs_param_unc_num, "bs_param_unc_num");
    param_num_ = dlsym_cast(library_, &bs_param_num, "bs_param_num");
    log_density_gradient_ = dlsym_cast(library_, &bs_log_density_gradient,
                                       "bs_log_density_gradient");
    param_constrain_ =
        dlsym_cast(library_, &bs_param_constrain, "bs_param_constrain");

    char *err = nullptr;
    model_ptr_ = std::unique_ptr<bs_model, decltype(&bs_model_destruct)>(
        model_construct(data, seed, &err), model_destruct);

    if (!model_ptr_) {
      if (err) {
        std::string error_string(err);
        free_error_msg_(err);
        throw std::runtime_error(error_string);
      }
      throw std::runtime_error("Failed to construct model");
    }

    // temporary: we probably don't want to store the RNG in the model
    // due to thread safety concerns
    rng_ptr_ = std::unique_ptr<bs_rng, decltype(&bs_rng_destruct)>(
        rng_construct(seed, &err), rng_destruct);
    if (!rng_ptr_) {
      if (err) {
        std::string error_string(err);
        free_error_msg_(err);
        throw std::runtime_error(error_string);
      }
      throw std::runtime_error("Failed to construct RNG");
    }
  }

  int unconstrained_dimensions() const {
    return param_unc_num_(model_ptr_.get());
  }
  int constrained_dimensions() const {
    return param_num_(model_ptr_.get(), true, true);
  }

  template <typename M>
  void logp_grad(const M &x, double &logp, M &grad) const {
    grad.resizeLike(x);

    char *err = nullptr;
    auto start = std::chrono::high_resolution_clock::now();
    int ret = log_density_gradient_(model_ptr_.get(), true, true, x.data(),
                                    &logp, grad.data(), &err);
    auto end = std::chrono::high_resolution_clock::now();
    total_time += std::chrono::duration<double>(end - start).count();
    ++count;

    if (ret != 0) {
      if (err) {
        std::string error_string(err);
        free_error_msg_(err);
        throw std::runtime_error(error_string);
      }
      throw std::runtime_error("Failed to compute log density and gradient");
    }
  }

  template <typename In, typename Out>
  void constrain_draw(In &&in, Out &&out) const {
    char *err = nullptr;
    int ret = param_constrain_(model_ptr_.get(), true, true, in.data(),
                               out.data(), rng_ptr_.get(), &err);

    if (ret != 0) {
      if (err) {
        std::string error_string(err);
        free_error_msg_(err);
        throw std::runtime_error(error_string);
      }
      throw std::runtime_error("Failed to constrain draw");
    }
  }

 private:
  std::unique_ptr<void, dlclose_deleter> library_;
  std::unique_ptr<bs_model, decltype(&bs_model_destruct)> model_ptr_;
  std::unique_ptr<bs_rng, decltype(&bs_rng_destruct)> rng_ptr_;
  decltype(&bs_free_error_msg) free_error_msg_;
  decltype(&bs_param_unc_num) param_unc_num_;
  decltype(&bs_param_num) param_num_;
  decltype(&bs_log_density_gradient) log_density_gradient_;
  decltype(&bs_param_constrain) param_constrain_;
};

using S = double;
using VectorS = Eigen::Matrix<S, -1, 1>;
using MatrixS = Eigen::Matrix<S, -1, -1>;

enum class Sampler { Nuts, Walnuts };

template <Sampler U, typename G>
void test_nuts(const DynamicStanModel &model, const VectorS &theta_init,
               G &generator, int N, S step_size, S max_depth, S max_error,
               const VectorS &inv_mass) {
  total_time = 0.0;
  count = 0;
  std::cout << std::endl
            << "D = " << model.unconstrained_dimensions() << ";  N = " << N
            << ";  step_size = " << step_size << ";  max_depth = " << max_depth
            << ";  WALNUTS = " << (U == Sampler::Walnuts ? "true" : "false")
            << std::endl;

  auto logp = [&model](auto &&...args) { model.logp_grad(args...); };

  int M = model.constrained_dimensions();

  MatrixS draws(M, N);
  auto writer = [&model, &draws](int n, const VectorS &theta) {
    model.constrain_draw(theta, draws.col(n));
  };

  auto global_start = std::chrono::high_resolution_clock::now();
  if constexpr (U == Sampler::Walnuts) {
    walnuts::walnuts(generator, logp, inv_mass, step_size, max_depth, max_error,
                     theta_init, N, writer);
  } else if constexpr (U == Sampler::Nuts) {
    nuts::nuts(generator, logp, inv_mass, step_size, max_depth, theta_init, N,
               writer);
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

  for (int d = 0; d < std::min(M, 5); ++d) {
    auto mean = draws.row(d).mean();
    auto var = (draws.row(d).array() - mean).square().sum() / (N - 1);
    auto stddev = std::sqrt(var);
    std::cout << "dim " << d << ": mean = " << mean << ", stddev = " << stddev
              << "\n";
  }
  if (M > 5) {
    std::cout << "... elided " << (M - 5) << " dimensions ..." << std::endl;
  }
}

int main(int argc, char *argv[]) {
  int seed = 333456;
  int N = 5000;
  double step_size = 0.025;
  int max_depth = 10;
  double max_error = 0.2;  // 80% Metropolis, 45% Barker

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

  int D = model.unconstrained_dimensions();

  Eigen::VectorXd inv_mass = Eigen::VectorXd::Ones(D);
  std::mt19937 generator(seed);
  std::normal_distribution<double> std_normal(0.0, 1.0);
  Eigen::VectorXd theta_init(D);
  for (int i = 0; i < D; ++i) {
    theta_init(i) = std_normal(generator);
  }

  test_nuts<Sampler::Nuts>(model, theta_init, generator, N, step_size,
                           max_depth, max_error, inv_mass);
  test_nuts<Sampler::Walnuts>(model, theta_init, generator, N, step_size,
                              max_depth, max_error, inv_mass);

  return 0;
}
