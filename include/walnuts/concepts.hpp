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

/**
 * @brief Concept for a Markov chain sampler.
 *
 * A type `S` satisfies `Sampler` if it provides:
 *  - `s()` callable on a non-const instance, returning a log density value
 *    convertible to `double`. Each call advances the chain by one
 *    iteration and returns the log density of the new draw. The
 *    sampler is responsible for delivering the draw itself to any
 *    chain-local handler it holds.
 *  - `s.dim()` callable on a const instance, returning a value
 *    convertible to `std::size_t`. This reports the dimensionality
 *    of the parameter space being sampled.
 *
 * The single-argument operator must be invocable through a
 * `std::reference_wrapper<S>`, which means `operator()` must not be
 * `const`-only — it should be a non-const member function (or
 * mutable-callable in some other way), since sampling advances
 * internal state such as the position, RNG, and any adapted
 * parameters.
 */
template <typename S>
concept Sampler = requires(S& s, const S& cs) {
    { s() } -> std::convertible_to<double>;
    { cs.dim() } -> std::convertible_to<std::size_t>;
};

/**
 * @brief Concept for a handler of cross-chain events.
 *
 * A type `H` satisfies `Handler` if it provides:
 *  - `on_r_hat(double)` callable on a non-const instance, returning `void`,
 *  - `r_hats()` callable on a const instance, returning a reference to a
 *    `std::vector<double>`.
 *
 * The `r_hats()` accessor must be callable on a const instance so that
 * consumers with only const access to the handler can read the recorded
 * values.
 */
template <typename H>
concept GlobalHandler = requires(H& h, const H& ch, double r_hat) {
  { h.on_r_hat(r_hat) } -> std::same_as<void>;
  { ch.r_hats() } -> std::same_as<const std::vector<double>&>;
};

/**
 * @brief Concept for a handler of chain-specific events.
 *
 * A type `C` satisfies `ChainHandler` if it provides the following
 * member functions, each callable on a non-const instance and returning
 * `void`:
 *  - `on_warmup(const Eigen::VectorXd&, double, double, const Eigen::VectorXd&)`
 *    called once per warmup draw with the position, log density, step
 *    size, and diagonal inverse mass matrix.
 *  - `on_warmup_complete(double, const Eigen::VectorXd&)` called once
 *    when warmup finishes, with the final step size and diagonal inverse
 *    mass matrix.
 *  - `on_sample(const Eigen::VectorXd&, double)` called once per
 *    post-warmup draw with the position and log density.
 *  - `on_stop()` called when sampling stops.
 */
template <typename C>
concept ChainHandler = requires(C& c,
                                const Eigen::VectorXd& position,
                                const Eigen::VectorXd& diag_inv_mass,
                                double lp,
                                double step_size) {
  { c.on_warmup(position, lp, step_size, diag_inv_mass) }
        -> std::same_as<void>;
  { c.on_warmup_complete(step_size, diag_inv_mass) }
        -> std::same_as<void>;
  { c.on_sample(position, lp) } -> std::same_as<void>;
  { c.on_stop() } -> std::same_as<void>;
};  
