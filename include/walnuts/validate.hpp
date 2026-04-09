#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace walnuts {

  template <typename T1, typename T2>
  void validate_same_size(const T1& x1, const T2& x2,
			  const std::string& fun,
			  const std::string& arg1,
			  const std::string& arg2) {
    if (x1.size() != x2.size()) {
      throw std::invalid_argument(fun + ": " + arg1 + " and " + arg2 + " must be same size");
    }
  }

  template <typename F, typename T>
  void validate_elements(const F& f,
			 T x,
			 std::string fun,
			 std::string arg,
			 std::string err) {
    if (!f(x)) {
      std::string msg = fun + " argument " + arg + " " + err;
      throw std::invalid_argument(msg);
    }
  }

  template <typename F, typename T, int M, int N>
  void validate_elements(const F& f,
			 const Eigen::Matrix<T, M, N>& x,
			 const std::string& fun,
			 const std::string& arg,
			 const std::string& err) {
    for (int64_t n = 0; n < x.size(); ++n) {
      validate_elements(f, x(n), fun, arg, err);
    }
  }
  
  template <typename T, typename F>
  void validate_elements(const F& test,
			 const std::vector<T>& xs,
			 const std::string& fun,
			 const std::string& arg,
			 const std::string& err) {
    for (const T& x : xs) {
      validate_elements(test, x, fun, arg, err);
    }
  }
  

  template <typename T>
  void validate_positive(T x, std::string fun, std::string arg) {
    auto is_pos = [](auto u) { return u > 0; };
    validate_elements(is_pos, x, fun, arg, "must be positive");
  }
}
