#pragma once

#include <random>
#include <ranges>
#include <type_traits>

#include <Eigen/Dense>

/**
 * @brief Concept for a type with a `.size()` member function.
 *
 * A type `T` satisfies `Sized` if `t.size()` is callable on a const
 * instance and returns a value comparable for equality. This matches
 * standard containers, `std::string`, `std::span`, and Eigen vectors
 * and matrices.
 */
template <typename T>
concept Sized = requires(const T& t) {
    { t.size() } -> std::equality_comparable;
};

/**
 * @brief Concept for an Eigen dense type with floating-point scalar.
 *
 * A type `T` satisfies `EigenDenseType` if it provides:
 *  - a nested `Scalar` type that is a floating-point type,
 *  - a `size()` member returning a value convertible to `Eigen::Index`,
 *  - element access via `t(i)` returning a value convertible to `Scalar`.
 *
 * Models include `Eigen::VectorXd`, `Eigen::MatrixXd`, and other
 * fixed- or dynamic-size Eigen dense types with floating-point scalars.
 * For matrices, `t(i)` accesses elements in storage order (column-major
 * by default).
 */
template <typename T>
concept EigenDenseType = requires(const T& t) {
    typename T::Scalar;
    { t.size() } -> std::convertible_to<Eigen::Index>;
    { t(0) } -> std::convertible_to<typename T::Scalar>;
} && std::floating_point<typename T::Scalar>;

/**
 * @brief Concept for a leaf type in the floating-point validation tree.
 *
 * A type `T` satisfies `FloatingPointLeaf` if it is either a
 * floating-point type or an Eigen dense type with floating-point
 * scalar. These are the types whose elements can be checked directly,
 * without further recursive descent through containers.
 */
template <typename T>
concept FloatingPointLeaf = std::floating_point<T> || EigenDenseType<T>;

/**
 * @brief Concept for a floating-point leaf or a one-level container
 * of such leaves.
 *
 * A type `T` satisfies `NestedFloatingPoint` if any of the following
 * hold:
 *  - `T` satisfies `FloatingPointLeaf` (a floating-point value or an
 *    Eigen dense type with floating-point scalar),
 *  - `T` is a `std::ranges::range` whose element type satisfies
 *    `FloatingPointLeaf`.
 *
 * This covers `double`, `Eigen::VectorXd`, `std::vector<double>`, and
 * `std::vector<Eigen::VectorXd>`, but does not recurse to deeper
 * nesting such as `std::vector<std::vector<double>>`.
 */
template <typename T>
concept NestedFloatingPoint =
    FloatingPointLeaf<T> ||
    (std::ranges::range<T> &&
     FloatingPointLeaf<std::ranges::range_value_t<T>>);
