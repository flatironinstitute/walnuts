// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Kalshi DPDK Trading System

#pragma once
#include <Eigen/Dense>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <Eigen/Dense>
/**
 * @file concepts.hpp
 * @brief C++20 concepts for compile-time interface checking
 *
 * @details This file defines concepts used across the trading system to enforce
 * interface contracts at compile time. This provides zero-cost abstraction
 * compared to runtime polymorphism via virtual functions.
 *
 * Design principles:
 * - Concepts enforce interface contracts without runtime overhead
 * - Used in template constraints to catch errors early
 * - Enable better compiler diagnostics than SFINAE
 */

namespace nuts {

//==============================================================================
// Integer Type Concepts
//==============================================================================

/// @brief Concept for 32-bit signed integers
template <typename T>
concept SignedInteger32 =
    sizeof(std::remove_cvref_t<T>) == 4 && std::is_signed_v<std::remove_cvref_t<T>> && std::is_integral_v<std::remove_cvref_t<T>>;

/// @brief Concept for 32-bit unsigned integers
template <typename T>
concept UnsignedInteger32 =
    sizeof(std::remove_cvref_t<T>) == 4 && !std::is_signed_v<std::remove_cvref_t<T>> && std::is_integral_v<std::remove_cvref_t<T>>;

/// @brief Concept for any 32-bit integer (signed or unsigned)
template <typename T>
concept Integer32 = sizeof(std::remove_cvref_t<T>) == 4 && std::is_integral_v<std::remove_cvref_t<T>>;

/// @brief Concept for 64-bit signed integers
template <typename T>
concept SignedInteger64 =
    sizeof(std::remove_cvref_t<T>) == 8 && std::is_signed_v<std::remove_cvref_t<T>> && std::is_integral_v<std::remove_cvref_t<T>>;

/// @brief Concept for 64-bit unsigned integers
template <typename T>
concept UnsignedInteger64 =
    sizeof(std::remove_cvref_t<T>) == 8 && !std::is_signed_v<std::remove_cvref_t<T>> && std::is_integral_v<std::remove_cvref_t<T>>;

/// @brief Concept for any 64-bit integer (signed or unsigned)
template <typename T>
concept Integer64 = sizeof(std::remove_cvref_t<T>) == 8 && std::is_integral_v<std::remove_cvref_t<T>>;

/// @brief Concept for 32 bit floating point
template <typename T>
concept FloatingPoint32 =
    sizeof(std::remove_cvref_t<T>) == 4 && std::is_floating_point_v<std::remove_cvref_t<T>>;

/// @brief Concept for 64 bit floating point
template <typename T>
concept FloatingPoint64 =
    sizeof(std::remove_cvref_t<T>) == 8 && std::is_floating_point_v<std::remove_cvref_t<T>>;

/// @brief Concept for any floating point type
template <typename T>
concept FloatingPoint = std::is_floating_point_v<std::remove_cvref_t<T>>;

/// @brief Concept for any integral type
template <typename T>
concept Integer = std::is_integral_v<std::remove_cvref_t<T>>;

namespace details {
/**
 * Checks if a type's pointer is convertible to a templated base type's pointer.
 * If the arbitrary function
 *
 * std::true_type f(const Base<Derived>*) *
 * is well formed for input `std::declval<Derived*>() this has a member
 *  value equal to `true`, otherwise the value is false.
 * @tparam Base The templated base type for valid pointer conversion.
 * @tparam Derived The type to check
 * @ingroup type_trait
 */
template <template <typename> class Base, typename Derived>
struct is_base_pointer_convertible {
  static std::false_type f(const void *);
  template <typename OtherDerived>
  static std::true_type f(const Base<OtherDerived> *);
  enum {
    value
    = decltype(f(std::declval<std::remove_reference_t<Derived> *>()))::value
  };
};
}

template <template <class...> class Base, typename Derived>
struct is_base_pointer_convertible : std::bool_constant<details::is_base_pointer_convertible<Base, Derived>::value> {};

template <template <class...> class Base, typename Derived>
inline bool constexpr is_base_pointer_convertible_v = is_base_pointer_convertible<Base, Derived>::value;

// “Any Eigen expression” (dense or sparse): derived from EigenBase
template <typename T>
concept EigenAny
    = is_base_pointer_convertible_v<Eigen::EigenBase, std::decay_t<T>>;

// Dense matrix/vector expressions (excludes ArrayBase): derived from MatrixBase
template <typename T>
concept EigenMatrixBase = is_base_pointer_convertible_v<Eigen::MatrixBase, std::decay_t<T>>;

// Dense objects including arrays: derived from DenseBase
template <typename T>
concept EigenDenseBase = is_base_pointer_convertible_v<Eigen::DenseBase, std::decay_t<T>>;

// 1D (vector) using Eigen compile-time shape
template <typename T>
concept EigenVector
    = EigenMatrixBase<T>
      && ((std::decay_t<T>::RowsAtCompileTime == 1 && std::decay_t<T>::ColsAtCompileTime != 1) ||
          (std::decay_t<T>::ColsAtCompileTime == 1 && std::decay_t<T>::RowsAtCompileTime != 1));

// Concept for a 1D column vector
template <typename T>
concept EigenColVector
    = EigenVector<T> && (std::decay_t<T>::ColsAtCompileTime == 1 && std::decay_t<T>::RowsAtCompileTime != 1);

// Concept for a 1d row vector
template <typename T>
concept EigenRowVector
    = EigenVector<T> && (std::decay_t<T>::RowsAtCompileTime == 1 && std::decay_t<T>::ColsAtCompileTime != 1);

// A 2d matrix
template <typename T>
concept EigenMatrix
    = EigenMatrixBase<T>
      && (std::decay_t<T>::RowsAtCompileTime != 1 && std::decay_t<T>::ColsAtCompileTime != 1);

// Concept for a log density/gradient function
template <typename F, typename Arg1, typename Arg2, typename Arg3>
concept LogDensityGradientFunction = EigenVector<Arg1> && FloatingPoint<Arg2> && EigenVector<Arg3> &&
  requires(F&& f, Arg1&& a1, Arg2&& a2, Arg3&& a3) {
    { std::forward<F>(f)(std::forward<Arg1>(a1), std::forward<Arg2>(a2), std::forward<Arg3>(a3)) } -> std::same_as<void>;
};

template <typename F, typename S>
concept LogGradFun = FloatingPoint<S> && LogDensityGradientFunction<F, Eigen::Matrix<S, -1, 1>&, S&, Eigen::Matrix<S, -1, 1>&>;


}  // namespace nuts
