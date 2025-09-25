#ifndef WALNUTS_LOAD_STAN_HPP
#define WALNUTS_LOAD_STAN_HPP

#include <bridgestan.h>

#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// TODO: consider using something like https://github.com/martin-olivier/dylib/
#ifdef _WIN32
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

struct dlclose_deleter {
  void operator()(void* handle) const {
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

class DynamicStanModel {
 public:
  DynamicStanModel(const char* model_path, const char* data, unsigned int seed)
      : library_(dlopen_safe(model_path)),
        model_ptr_(make_model(library_, data, seed)),
        free_error_msg_(dlsym_cast(library_, bs_free_error_msg)),
        param_unc_num_(dlsym_cast(library_, bs_param_unc_num)),
        param_num_(dlsym_cast(library_, bs_param_num)),
        log_density_gradient_(dlsym_cast(library_, bs_log_density_gradient)),
        param_constrain_(dlsym_cast(library_, bs_param_constrain)),
        param_names_(dlsym_cast(library_, bs_param_names)),
        rng_ptr_(nullptr, [](auto) {}) {
    // temporary: we probably don't want to store the RNG in the model
    // due to thread safety concerns
    auto rng_construct = dlsym_cast(library_, bs_rng_construct);
    auto rng_destruct = dlsym_cast(library_, bs_rng_destruct);
    char* err = nullptr;
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
  void constrain_draw(In&& in, Out&& out) const {
    char* err = nullptr;
    int ret = param_constrain_(model_ptr_.get(), true, true, in.data(),
                               out.data(), rng_ptr_.get(), &err);

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

 private:
  dynamic_library library_;
  unique_bs_model model_ptr_;
  decltype(&bs_free_error_msg) free_error_msg_;
  decltype(&bs_param_unc_num) param_unc_num_;
  decltype(&bs_param_num) param_num_;
  decltype(&bs_log_density_gradient) log_density_gradient_;
  decltype(&bs_param_constrain) param_constrain_;
  decltype(&bs_param_names) param_names_;
  std::unique_ptr<bs_rng, decltype(&bs_rng_destruct)> rng_ptr_;
};

#endif
