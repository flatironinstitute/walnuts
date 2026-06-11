#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "walnuts/concepts.hpp"

namespace walnuts::detail {

/**
 * @brief Validate that the specified stream is open.
 *
 * @tparam S The type of the stream.
 * @param[in] s The stream.
 * @param[in] name The name of the stream.
 * @throw invalid_argument If the stream is not open.
 */
template <OpenableStream S>
inline void validate_open(const S& s, const std::string& name) {
  if (s.is_open()) {
    return;
  }
  throw std::invalid_argument("could not open stream from: " + name);
}

/**
 * @brief Validate the standard vector is the specified size.
 *
 * @tparam T The type of vector elements.
 * @param[in] x The vector.
 * @param[in] size The size to test.
 * @param[in] var The name of the vector.
 * @param[in] target The name of size.
 * @throw invalid_argument If the container size is not the specified size.
 */
template <typename T>
inline void validate_size(const std::vector<T>& x, std::size_t size,
                          const std::string& var, const std::string& target) {
  if (x.size() == size) {
    return;
  }
  throw std::invalid_argument(var + " size must match " + target);
}

/**
 * @brief Validate the Eigen container is the specified size.
 *
 * @tparam T The type of vector elements.
 * @tparam R The size or dynamic specification of rows.
 * @tparam C The size or dynamic specification of columns.
 * @param[in] x The container.
 * @param[in] size The size to test.
 * @param[in] var The name of the container.
 * @param[in] target The name of size.
 * @throw invalid_argument If the container size is not the specified size.
 */
template <std::floating_point T, int R, int C>
inline void validate_size(const Eigen::Matrix<T, R, C>& x, std::size_t size,
                          const std::string& var, const std::string& target) {
  if (x.size() == static_cast<Eigen::Index>(size)) {
    return;
  }
  throw std::invalid_argument(var + " size must match " + target);
}

/**
 * @brief Throw an exception if the containers do not have the same size.
 *
 * @tparam T1 Type of first container.
 * @tparam T2 Type of second container.
 * @param[in] x1 The first container.
 * @param[in] x2 The second container.
 * @param[in] name1 The first container's name.
 * @param[in] name2 The second container's name.
 * @throw std::invalid_argument If the containers are not the same size.
 */
template <Sized T1, Sized T2>
inline void validate_same_size(const T1& x1, const T2& x2,
                               const std::string& name1,
                               const std::string& name2) {
  if (x1.size() == x2.size()) {
    return;
  }
  std::string msg = name1 + " and " + name2 + " must be the same size.";
  throw std::invalid_argument(msg);
}

/**
 * @brief Throw an exception if the value is not finite and > 1.
 *
 * @tparam T The type of value.
 * @param[in] x The value.
 * @param[in] var The name of the value.
 * @throw std::invalid_argument If the value is not finite and > 1.
 */
template <std::floating_point T>
inline void validate_finite_gt1(T x, const std::string& var) {
  if (std::isfinite(x) && x > 1) {
    return;
  }
  throw std::invalid_argument(var + " must be finite and > 1");
}

/**
 * @brief Throw an exception if the value is not finite and > 0.
 *
 * @tparam T The type of value.
 * @param[in] x The value.
 * @param[in] var The name of the value.
 * @throw std::invalid_argument If the value is not finite and > 0.
 */
template <std::floating_point T>
inline void validate_finite_positive(T x, const std::string& var) {
  if (std::isfinite(x) && x > 0) {
    return;
  }
  throw std::invalid_argument(var + " must be finite and > 0");
}

/**
 * @brief Throw an exception if the container's elements are not
 * finite and > 0.
 *
 * @tparam T The type of values.
 * @tparam R The row size (or -1 for dynamic).
 * @tparam C The column size (or -1 for dynamic).
 * @param[in] xs The container.
 * @param[in] var The name of the container.
 * @throw std::invalid_argument If the container has an element that
 * is not finite and > 0.
 */
template <std::floating_point T, int R, int C>
inline void validate_finite_positive(const Eigen::Matrix<T, R, C>& xs,
                                     const std::string& var) {
  for (Eigen::Index i = 0; i < xs.size(); ++i) {
    validate_finite_positive(xs(i), var);
  }
}

/**
 * @brief Throw an exception if the container's elements are not
 * finite and > 0.
 *
 * @tparam T The type of values (container or floating point).
 * @param[in] xs The container.
 * @param[in] var The name of the container.
 * @throw std::invalid_argument If the container has an element that
 * is not finite and > 0.
 */
template <NestedFloatingPoint T>
inline void validate_finite_positive(const std::vector<T>& xs,
                                     const std::string& var) {
  for (const auto& x : xs) {
    validate_finite_positive(x, var);
  }
}

/**
 * @brief Throw an exception if the value is not finite.
 *
 * @tparam T The type of value.
 * @param[in] x The value.
 * @param[in] var The name of the value.
 * @throw std::invalid_argument If the value is not finite.
 */
template <std::floating_point T>
inline void validate_finite(T x, const std::string& var) {
  if (std::isfinite(x)) {
    return;
  }
  throw std::invalid_argument(var + " must be finite");
}

/**
 * @brief Throw an exception if the container's elements are not
 * finite.
 *
 * @tparam T The type of values.
 * @tparam R The row size (or -1 for dynamic).
 * @tparam C The column size (or -1 for dynamic).
 * @param[in] xs The container.
 * @param[in] var The name of the container.
 * @throw std::invalid_argument If the container has an element that
 * is not finite.
 */
template <std::floating_point T, int R, int C>
inline void validate_finite(const Eigen::Matrix<T, R, C>& xs,
                            const std::string& var) {
  for (Eigen::Index i = 0; i < xs.size(); ++i) {
    validate_finite(xs(i), var);
  }
}

/**
 * @brief Throw an exception if the container's elements are not
 * finite.
 *
 * @tparam T The type of values (container or floating point).
 * @param[in] xs The container.
 * @param[in] var The name of the container.
 * @throw std::invalid_argument If the container has an element that
 * is not finite.
 */
template <NestedFloatingPoint T>
inline void validate_finite(const std::vector<T>& xs, const std::string& var) {
  for (const auto& x : xs) {
    validate_finite(x, var);
  }
}

/**
 * @brief Throw an exception if the value is not positive and finite.
 *
 * @tparam T Type of value.
 * @param[in] x The variable's value.
 * @param[in] name The variable's name.
 * @throw std::invalid_argument If the value is not positive and finite.
 */
template <std::floating_point T>
inline void validate_positive(T x, const std::string& name) {
  if (x > 0 && !std::isinf(x)) {
    return;
  }
  std::string msg = name + " must be in (0, inf).";
  throw std::invalid_argument(msg);
}

/**
 * @brief Throw an exception if the value is not positive.
 *
 * @tparam T Type of integral value.
 * @param[in] x The integral value.
 * @param[in] name The name of the variable.
 * @throw std::invalid_argument If the value is not positive.
 */
template <std::integral T>
inline void validate_positive(T x, const std::string& name) {
  if (x > 0) {
    return;
  }
  std::string msg = name + " must be in {1, 2, ... }";
  throw std::invalid_argument(msg);
}

/**
 * @brief Throw an exception if the Eigen container's components are
 * not positive and finite.
 *
 * @tparam T The type of elements.
 * @param[in] x The Eigen container.
 * @param[in] name The container's name.
 * @throw std::invalid_argument If the elements of the container are
 * not all positive and finite.
 */
template <std::floating_point T, int R, int C>
inline void validate_positive(const Eigen::Matrix<T, R, C>& x,
                              const std::string& name) {
  if ((x.array() > 0.0).all() && x.allFinite()) {
    return;
  }
  std::string msg = name + " must be in (0, inf).";
  throw std::invalid_argument(msg);
}

/**
 * @brief Throw an exception if the value is not in (0, 1).
 *
 * @tparam T Type of value.
 * @param[in] x The variable's value.
 * @param[in] name The variable's name.
 * @throw std::invalid_argument If the value is not in (0, 1).
 */
template <std::floating_point T>
inline void validate_probability(T x, const std::string& name) {
  if (x > 0 && x < 1) {
    return;
  }
  std::string msg = name + " must be in (0, 1)";
  throw std::invalid_argument(msg);
}

/**
 * @brief Throw an exception if the value is not in [0, 1].
 *
 * @tparam T Type of value.
 * @param[in] x The variable's value.
 * @param[in] name The variable's name.
 * @throw std::invalid_argument If the value is not in [0, 1].
 */
template <std::floating_point T>
inline void validate_probability_inclusive(T x, const std::string& name) {
  if (x >= 0 && x <= 1) {
    return;
  }
  std::string msg = name + " must be in [0, 1]";
  throw std::invalid_argument(msg);
}

}  // namespace walnuts::detail
