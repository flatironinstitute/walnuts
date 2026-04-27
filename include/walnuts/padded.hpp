#pragma once

#include <array>
#include <cstddef>

namespace walnuts {

/**
 * @brief Go with a conservative constant destructive interference size.  
 *
 * The std::hardware_destructive_interference_size is not universally supported
 * and can underreport when it is supported.  128 is safe for ARM and Intel
 * hardware. 
 */
inline constexpr std::size_t CACHE_LINE_SIZE = 128;

/**
 * @brief A wrapper of `T` aligned to `CACHE_LINE_SIZE` and padded to minimum
 * multiple of `CACHE_LINE_SIZE` big enough to hold value.
 *
 * @tparam T The type of value.
 */
template <class T, std::size_t Align = CACHE_LINE_SIZE>
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
