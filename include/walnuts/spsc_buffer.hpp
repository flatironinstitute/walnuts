#pragma once

#include <array>
#include <atomic>
#include <concepts>
#include <cstdint>

#include "walnuts/util.hpp"

namespace walnuts::detail {

/**
 * @brief A single-producer, single-consumer buffer backed by a
 * lock-free triple buffer.
 *
 * Initialization: The buffer is initialized with an initial
 * value either explicitly or through the default constructor
 * for the value type and then made available to a single consumer
 * and to a single producer.
 *
 * Warning: This class is not safe for use by more than one consumer
 * or more than one producer.
 *
 * @code
 * Foo v_init;
 * SpscBuffer<Foo> buf(v_init);
 * @endcode
 *
 * Reading: `read_latest()` will always return the latest value
 * published.
 *
 * @code
 * Foo v_latest = buf.read_latest();
 * @endcode
 *
 * `read_latest()` is not generically thread safe and must only be
 * called by the single consumer.  It also only returns a value
 * that is valid until the next call to `read_latest()`.
 *
 * Writing: The write code will only be used by a single producer.
 * The following code writes a value into the buffer.
 *
 * @code
 * Foo v_to_write = ...;
 * Foo& v_buf = buf.write_buffer();
 * v_buf = v_to_write;
 * buf.publish();
 * @endcode
 *
 * `write_buffer()` and `publish()` are not generically thread safe
 * and must only be called by the single producer.
 *
 * Unlike a queue pattern, `publish()` overwrites the current value
 * even if the current value has never been read.
 *
 * Implementation note: The implementation uses bit twiddling,
 * packing both a dirty bit indicator and the value 0, 1, or 2, into
 * a single `uint32_t` in order to allow the swaps to be
 * atomic. Helper functions are provided to read the dirty bit, add
 * a dirty bit to an index, or return the index from a potentially
 * dirty bit.
 *
 * See @cite brilliantsugar2024triplebuffer.
 * brilliantsugar. 2024. How I learned to stop worrying and love
 * juggling C++ atomics.
 *
 * @tparam T Type of value buffered.
 */
template <typename T>
class SpscBuffer {
 public:
  /**
   * @brief Construct a buffer filled with copies of the specified value.
   *
   * @param[in] t Value to write.
   */
  explicit SpscBuffer(const T& t)
    requires std::copy_constructible<T>
      : buffers_{{{t}, {t}, {t}}} {}

  /**
   * @brief Construct a buffer filled with default-constructed values.
   */
  SpscBuffer()
    requires std::default_initializable<T>
      : SpscBuffer(T()) {}

  SpscBuffer(const SpscBuffer&) = delete;
  SpscBuffer& operator=(const SpscBuffer&) = delete;
  SpscBuffer(SpscBuffer&&) = delete;
  SpscBuffer& operator=(SpscBuffer&&) = delete;

  /**
   * @brief Return a reference to the latest value.
   *
   * @return A reference to the latest value.
   */
  [[nodiscard]] const T& read_latest() noexcept {
    std::uint32_t mid = middle_.load(std::memory_order_relaxed);
    if (is_dirty(mid)) {
      std::uint32_t prev = middle_.exchange(front_, std::memory_order_acq_rel);
      front_ = index_of(prev);
    }
    return buffers_[front_].data;
  }

  /**
   * @brief Return a reference to a value into which to write.
   *
   * After a value is written into the return, the `publish()` method must
   * be called to swap it in.
   *
   * @return Reference into which to write a value.
   */
  [[nodiscard]] T& write_buffer() noexcept { return buffers_[back_].data; }

  /**
   * @brief Commit the latest state of the write buffer.
   */
  void publish() noexcept {
    std::uint32_t prev =
        middle_.exchange(make_dirty(back_), std::memory_order_acq_rel);
    back_ = index_of(prev);
  }

 private:
  /** @brief Bit to mark an index as dirty */
  static constexpr std::uint32_t DIRTY_BIT = std::uint32_t{1} << 31;

  /** @brief Mask to return the index. */
  static constexpr std::uint32_t INDEX_MASK = ~DIRTY_BIT;

  /**
   * @brief Return the index marked as dirty.
   *
   * @see index_of to recover the underlying index.
   *
   * @param[in] idx Index to make dirty.
   * @return The dirty form of the index.
   */
  static constexpr std::uint32_t make_dirty(std::uint32_t idx) noexcept {
    return idx | DIRTY_BIT;
  }

  /**
   * @brief Return the clean form of the specified index.
   *
   * @param[in] idx A potentially dirty index.
   * @return The index underlying the argument in {0, 1, 2}.
   */
  static constexpr std::uint32_t index_of(std::uint32_t idx) noexcept {
    return idx & INDEX_MASK;
  }

  /**
   * @brief Return true if the index is dirty.
   *
   * @param[in] idx The index.
   * @return `true` if the index is dirty.
   */
  static constexpr bool is_dirty(std::uint32_t idx) noexcept {
    return (idx & DIRTY_BIT) != 0;
  }

  /**
   * @brief A structure to hold an aligned form of the type `T`.
   */
  struct alignas(FALSE_SHARING_GUARD_SIZE) AlignedT {
    T data;
  };

  std::array<AlignedT, 3> buffers_;
  std::atomic<std::uint32_t> middle_{1};
  alignas(FALSE_SHARING_GUARD_SIZE) std::uint32_t back_{0};
  alignas(FALSE_SHARING_GUARD_SIZE) std::uint32_t front_{2};
};

}  // namespace walnuts::detail
