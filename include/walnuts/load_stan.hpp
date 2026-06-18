#ifndef WALNUTS_LOAD_STAN_HPP
#define WALNUTS_LOAD_STAN_HPP

// TODO: not entirely happy with this file living in 'include/',
// but it is used by both the examples and python bindings

#include <bridgestan.h>
#include <Eigen/Dense>

#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>



// TODO: consider using something like https://github.com/martin-olivier/dylib/
#if defined _WIN32 || defined __MINGW32__
// hacky way to get dlopen and friends on Windows
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define dlopen(lib, flags) static_cast<void*>(LoadLibraryA(lib))
#define dlsym(handle, sym) GetProcAddress(static_cast<HMODULE>(handle), sym)
#define dlclose(handle) FreeLibrary(static_cast<HMODULE>(handle))

static char* dlerror() {
  DWORD err = GetLastError();
  int length = snprintf(NULL, 0, "%ld", err);
  char* str = static_cast<char*>(malloc(length + 1));
  snprintf(str, length + 1, "%ld", err);
  return str;
}

#else
#include <dlfcn.h>
#endif

namespace walnuts::internal {

struct dlclose_deleter {
  void operator()(void*) const {
    // TODO: Crashes on some systems, see
    // https://github.com/flatironinstitute/walnuts/pull/25#discussion_r2298576937
    // if (handle) {
    //   dlclose(handle);
    // }
  }
};

using dynamic_library = std::unique_ptr<void, dlclose_deleter>;

inline dynamic_library dlopen_safe(const char* path) {
  auto handle = dlopen(path, RTLD_NOW | RTLD_NODELETE);
  if (!handle) {
    throw std::runtime_error(std::string("Error loading library '") + path +
                             "': " + dlerror());
  }
  return dynamic_library(handle);
}

template <typename T>
inline T dlsym_cast_impl(dynamic_library& library, const char* name) {
  auto sym = dlsym(library.get(), name);
  if (!sym) {
    throw std::runtime_error(std::string("Error loading symbol '") + name +
                             "': " + dlerror());
  }
  return reinterpret_cast<T>(sym);
}

#define dlsym_cast(library, func) \
  dlsym_cast_impl<decltype(&func)>(library, #func)

using unique_bs_model = std::unique_ptr<bs_model, decltype(&bs_model_destruct)>;

inline unique_bs_model make_model(dynamic_library& library, const char* data,
                                  unsigned int seed) {
  auto model_construct = dlsym_cast(library, bs_model_construct);
  auto model_destruct = dlsym_cast(library, bs_model_destruct);
  char* err = nullptr;
  auto model_ptr =
      unique_bs_model(model_construct(data, seed, &err), model_destruct);
  if (!model_ptr) {
    if (err) {
      std::string error_string(err);
      dlsym_cast(library, bs_free_error_msg)(err);
      throw std::runtime_error(error_string);
    }
    throw std::runtime_error("Failed to construct model");
  }
  return model_ptr;
}

}  // namespace walnuts::internal

namespace walnuts {

using unique_bs_rng = std::unique_ptr<bs_rng, decltype(&bs_rng_destruct)>;

class DynamicStanModel {
 public:
  DynamicStanModel(const char* model_path, const char* data, unsigned int seed)
      : library_(internal::dlopen_safe(model_path)),
        model_ptr_(internal::make_model(library_, data, seed)),
        free_error_msg_(dlsym_cast(library_, bs_free_error_msg)),
        param_unc_num_(dlsym_cast(library_, bs_param_unc_num)),
        param_num_(dlsym_cast(library_, bs_param_num)),
        log_density_gradient_(dlsym_cast(library_, bs_log_density_gradient)),
        param_constrain_(dlsym_cast(library_, bs_param_constrain)),
        param_initialize_(dlsym_cast(library_, bs_param_initialize)),
        param_names_(dlsym_cast(library_, bs_param_names)),
        rng_construct_(dlsym_cast(library_, bs_rng_construct)),
        rng_destruct_(dlsym_cast(library_, bs_rng_destruct)) {}

  std::size_t unconstrained_dimensions() const {
    return static_cast<std::size_t>(param_unc_num_(model_ptr_.get()));
  }
  std::size_t constrained_dimensions() const {
    return static_cast<std::size_t>(param_num_(model_ptr_.get(), true, true));
  }

  template <typename M>
  inline void logp_grad(const M& x, double& logp, M& grad) const {
    grad.resizeLike(x);

    char* err = nullptr;
    int ret = log_density_gradient_(model_ptr_.get(), true, true, x.data(),
                                    &logp, grad.data(), &err);

    if (ret != 0) {
      if (err) {
        std::string error_string(err);
        free_error_msg_(err);
        std::cerr << "Error in logp_grad: " << error_string << std::endl;

        logp = -std::numeric_limits<double>::infinity();
        grad.setZero();
        return;
      }
      throw std::runtime_error("Failed to compute log density and gradient");
    }
  }

  template <typename In, typename Out>
  void constrain_draw(In&& in, Out&& out, unique_bs_rng& rng) const {
    char* err = nullptr;
    int ret = param_constrain_(model_ptr_.get(), true, true, in.data(),
                               out.data(), rng.get(), &err);

    if (ret != 0) {
      if (err) {
        std::string error_string(err);
        free_error_msg_(err);
        std::cerr << "Error in constrain_draw: " << error_string << std::endl;
        out.array() = std::numeric_limits<double>::quiet_NaN();
        return;
      }
      throw std::runtime_error("Failed to constrain draw");
    }
  }

  Eigen::VectorXd initialize(const char* json, unique_bs_rng& rng,
                             double init_radius) const {
    Eigen::VectorXd params(unconstrained_dimensions());
    char* err = nullptr;
    int ret = param_initialize_(model_ptr_.get(), json, rng.get(), init_radius,
                                100, true, params.data(), &err);
    if (ret != 0) {
      if (err) {
        std::string error_string(err);
        free_error_msg_(err);
        throw std::runtime_error(error_string);
      }
      throw std::runtime_error("Failed to initialize model");
    }
    return params;
  }

  std::vector<std::string> param_names() const {
    std::vector<std::string> names;
    names.reserve(constrained_dimensions());

    const char* csv_names = param_names_(model_ptr_.get(), true, true);
    const char* p;
    for (p = csv_names; *p != '\0'; ++p) {
      if (*p == ',') {
        names.emplace_back(csv_names, p - csv_names);
        csv_names = p + 1;
      }
    }
    names.emplace_back(csv_names, p - csv_names);

    return names;
  }

  unique_bs_rng make_rng(unsigned int seed) const {
    char* err = nullptr;
    auto rng = unique_bs_rng(rng_construct_(seed, &err), rng_destruct_);
    if (!rng) {
      if (err) {
        std::string error_string(err);
        free_error_msg_(err);
        throw std::runtime_error(error_string);
      }
      throw std::runtime_error("Failed to construct RNG");
    }

    return rng;
  }

 private:
  internal::dynamic_library library_;
  internal::unique_bs_model model_ptr_;
  decltype(&bs_free_error_msg) free_error_msg_;
  decltype(&bs_param_unc_num) param_unc_num_;
  decltype(&bs_param_num) param_num_;
  decltype(&bs_log_density_gradient) log_density_gradient_;
  decltype(&bs_param_constrain) param_constrain_;
  decltype(&bs_param_initialize) param_initialize_;
  decltype(&bs_param_names) param_names_;
  decltype(&bs_rng_construct) rng_construct_;
  decltype(&bs_rng_destruct) rng_destruct_;
};

}  // namespace walnuts

// macro clean up
#undef dlsym_cast
#if defined _WIN32 || defined __MINGW32__
#undef WIN32_LEAN_AND_MEAN
#undef dlopen
#undef dlsym
#undef dlclose
#endif

#endif
