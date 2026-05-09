#pragma once

#include <array>
#include <cstddef>

#include <walnuts/util.hpp>

namespace walnuts {

/**
 * @brief A wrapper of `T` aligned to `FALSE_SHARING_GUARD_SIZE`.
 *
 * Warning: the padding bytes after `T` are uninitialized and
 * therefore should not be used with `memcmp` or be serialized (to
 * avoid data leakage from the heap).
 *
 * @tparam T The type of value.
 */
template <class T, std::size_t Align = FALSE_SHARING_GUARD_SIZE>
struct alignas(Align) Padded {
  static_assert((Align & (Align - 1)) == 0, "Align must be a power of two");
  static_assert(Align >= alignof(T), "Align must be >= alignof(T)");

  /**
   * @brief The value.
   */
  T val{};
};

}  // namespace walnuts
