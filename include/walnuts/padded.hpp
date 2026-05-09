#pragma once

#include <array>
#include <cstddef>

#include <walnuts/util.hpp>

namespace walnuts {

/**
 * @brief A wrapper of `T` aligned to `FALSE_SHARING_GUARD_SIZE` and
 * padded to minimum multiple of `FALSE_SHARING_GUARD_SIZE` big enough
 * to hold value.
 *
 * @tparam T The type of value.
 */
template <class T, std::size_t Align = FALSE_SHARING_GUARD_SIZE>
struct alignas(Align) Padded {
  static_assert((Align & (Align - 1)) == 0, "Align must be a power of two");
  static_assert(Align >= alignof(T), "Align must be >= alignof(T)");

  /**
   * @brief Padding bytes required to make `sizeof(Padded<T>)` at least
   * `Align` and a multiple of `Align`.
   */
  static constexpr std::size_t PADDING_BYTES =
      (Align - (sizeof(T) % Align)) % Align;

  /**
   * @brief The value.
   */
  T val{};

  /**
   * @brief The unused padding bytes.
   */
  std::array<std::byte, PADDING_BYTES> pad{};
};

}  // namespace walnuts
