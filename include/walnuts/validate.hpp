#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace nuts {

  template <typename T1, typename T2>
  inline void validate_same_size(const T1& x1, const T2& x2,
                                 const std::string& fun,
                                 const std::string& arg1,
                                 const std::string& arg2) {
    if (x1.size() != x2.size()) {
      throw std::invalid_argument(fun + ": " + arg1 + " and " + arg2 + " must be same size");
    }
  }

  template <typename F, typename T>
  inline void validate_elements(const F& f,
                                T x,
                                const std::string& fun,
                                const std::string& arg,
                                const std::string& err) {
    if (!f(x)) {
      std::string msg = fun + " argument " + arg + " " + err;
      throw std::invalid_argument(msg);
    }
  }

  template <typename F, typename T, int M, int N>
  inline void validate_elements(const F& f,
                                const Eigen::Matrix<T, M, N>& x,
                                const std::string& fun,
                                const std::string& arg,
                                const std::string& err) {
    for (int64_t n = 0; n < x.size(); ++n) {
      validate_elements(f, x(n), fun, arg, err);
    }
  }
  
  template <typename F, typename T>
  inline void validate_elements(const F& test,
                                const std::vector<T>& xs,
                                const std::string& fun,
                                const std::string& arg,
                                const std::string& err) {
    for (const T& x : xs) {
      validate_elements(test, x, fun, arg, err);
    }
  }
  

  template <typename T>
  inline void validate_positive(T x,
                                const std::string& fun,
                                const std::string& arg) {
    auto is_pos = [](auto u) { return u > 0; };
    validate_elements(is_pos, x, fun, arg, "must be positive");
  }


  template <std::floating_point T>
  inline void validate_probability(T p, const std::string& var) {
    if (!(p > 0 && p < 1))
      throw std::invalid_argument(var + " must be a proportion in (0, 1)");
  }
  
  template <typename T>
  inline void validate_size(const std::vector<T>& x, uint64_t size,
                            const std::string& var, const std::string& target) {
    if (x.size() != size)
      throw std::invalid_argument(var + " size must match " + target);
  }

  template <typename T, int R, int C>
  inline void validate_size(const Eigen::Matrix<T, R, C>& x, uint64_t size,
                            const std::string& var, const std::string& target) {
    if (x.size() != static_cast<int64_t>(size))
      throw std::invalid_argument(var + " size must match " + target);
  }

  template <typename T>
  inline void validate_gt0(T x, const std::string& var) {
    if (!(x > 0))
      throw std::invalid_argument(var + " must be > 0");
  }

  template <std::floating_point T>
  inline void validate_finite_gt1(T x, const std::string& var) {
    if (!(std::isfinite(x) && x > 1))
      throw std::invalid_argument(var + " must be finite and > 1");
  }

  template <std::integral T>
  inline void validate_finite_positive(T x, const std::string& var) {
    if (!(x > 0))
      throw std::invalid_argument(var + " must be > 0");
  }

  template <std::floating_point T>
  inline void validate_finite_positive(T x, const std::string& var) {
    if (!(std::isfinite(x) && x > 0))
      throw std::invalid_argument(var + " must be finite and > 0");
  }
  
  template <typename T, int R, int C>
  inline void validate_finite_positive(const Eigen::Matrix<T, R, C>& xs, const std::string& var) {
    std::string var_entries = var + " entries";
    for (Eigen::Index i = 0; i < xs.size(); ++i)
      validate_finite_positive(xs(i), var_entries);
  }

  template <typename T>
  inline void validate_finite_positive(const std::vector<T>& xs, const std::string& var) {
    std::string var_entries = var + " entries";
    for (const auto& x : xs)
      validate_finite_positive(x, var_entries);
  }

  template <std::floating_point T>
  inline void validate_finite(T x, const std::string& var) {
    if (!std::isfinite(x))
      throw std::invalid_argument(var + " must be finite");
  }
  
  template <typename T, int R, int C>
  inline void validate_finite(const Eigen::Matrix<T, R, C>& xs, const std::string& var) {
    std::string var_entries = var + " entries";
    for (int64_t i = 0; i < xs.size(); ++i)
      validate_finite(xs(i), var_entries);
  }

  template <typename T>
  inline void validate_finite(const std::vector<T>& xs, const std::string& var) {
    std::string var_entries = var + " entries";
    for (const auto& x : xs)
      validate_finite(x, var_entries);
  }
  
  template <class Stream>
  inline void validate_open(const Stream& s, const std::string& name) {
    if (!s.is_open()) {
      throw std::invalid_argument("could not open stream from: " + name);
    }
  }
}
