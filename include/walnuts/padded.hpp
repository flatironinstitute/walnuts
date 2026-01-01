#pragma once

#include <array>
#include <cstddef>
#include <new>

namespace walnuts {

/**
 * @brief The destructive interference size, if non-zero, else 128.
 */
static constexpr std::size_t DI_SIZE =
    std::hardware_destructive_interference_size > 0
        ? std::hardware_destructive_interference_size
        : 128;

/**
 * @brief A wrapper of `T` aligned to `DI_SIZE` and padded to multiple
 * of `DI_SIZE`.
 *
 * @tparam T The type of value.
 */
template <class T>
struct alignas(DI_SIZE) Padded {
  /**
   * @brief Padding bytes required to make sizeof(Padded<T>) at least
   * DI_SIZE and a multiple of DI_SIZE.
   */
  static constexpr std::size_t PADDING_BYTES =
    (DI_SIZE - (sizeof(T) % DI_SIZE)) % DI_SIZE;

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
