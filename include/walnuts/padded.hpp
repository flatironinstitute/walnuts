// include/walnuts/padded.hpp

#pragma once

#include <array>
#include <cstddef>
#include <new>

namespace walnuts {

/**
 * @brief The destructive interference size, if non-zero, else 128.
 */
inline constexpr std::size_t DI_SIZE =
    std::hardware_destructive_interference_size > 0
        ? std::hardware_destructive_interference_size
        : 128;

/**
 * @brief A wrapper of `T` aligned to `DI_SIZE` and padded to minimum
 * multiple of `DI_SIZE` big enough to hold value.
 *
 * @tparam T The type of value.
 */
template <class T, std::size_t Align = DI_SIZE>
struct alignas(Align) Padded {
  /**
   * @brief Padding bytes required to make `sizeof(Padded<T>)` at least
   * `Align` and a multiple of `Align`.
   */
  static constexpr std::size_t PADDING_BYTES =
    (Align - (sizeof(T) % Align)) % Align;

  /**
   * @brief The value.
   */
  T val;

  /**
   * @brief The unused padding bytes.
   */
  std::array<std::byte, PADDING_BYTES> pad_{};
};

}  // namespace walnuts
