#ifndef WALNUTS_LOAD_STAN_HPP
#define WALNUTS_LOAD_STAN_HPP

#include <bridgestan.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// TODO: consider using something like https://github.com/martin-olivier/dylib/
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

inline auto dlopen_safe(const char *path) {
  auto handle = dlopen(path, RTLD_NOW);
  if (!handle) {
    throw std::runtime_error(std::string("Error loading library '") + path +
                             "': " + dlerror());
  }
  return std::unique_ptr<void, dlclose_deleter>(handle);
}

template <typename U, typename T>
inline auto dlsym_cast(U &library, T &&, const char *name) {
  auto sym = dlsym(library.get(), name);
  if (!sym) {
    throw std::runtime_error(std::string("Error loading symbol '") + name +
                             "': " + dlerror());
  }
  return reinterpret_cast<T>(sym);
}

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
    param_names_ = dlsym_cast(library_, &bs_param_names, "bs_param_names");

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
  inline void logp_grad(const M &x, double &logp, M &grad) const {
    grad.resizeLike(x);

    char *err = nullptr;
    int ret = log_density_gradient_(model_ptr_.get(), true, true, x.data(),
                                    &logp, grad.data(), &err);

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

  std::vector<std::string> param_names() const {
    std::vector<std::string> names;
    names.reserve(constrained_dimensions());

    const char *csv_names = param_names_(model_ptr_.get(), true, true);
    const char *p;
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
  std::unique_ptr<void, dlclose_deleter> library_;
  std::unique_ptr<bs_model, decltype(&bs_model_destruct)> model_ptr_;
  std::unique_ptr<bs_rng, decltype(&bs_rng_destruct)> rng_ptr_;
  decltype(&bs_free_error_msg) free_error_msg_;
  decltype(&bs_param_unc_num) param_unc_num_;
  decltype(&bs_param_num) param_num_;
  decltype(&bs_log_density_gradient) log_density_gradient_;
  decltype(&bs_param_constrain) param_constrain_;
  decltype(&bs_param_names) param_names_;
};

#endif
