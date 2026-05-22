#pragma once

#include <concepts>
#include <cstddef>
#include <mutex>
#include <type_traits>
#include <utility>

#include <walnuts/concepts.hpp>
#include <walnuts/util.hpp>

namespace walnuts {
namespace detail {

  /**
   * @brief A single-producer, single-consumer queue backed by a triple buffer.
   *
   * In this buffer, `read_latest()` will always return the latest
   * version published.  The `write_buffer()` and `publish()` pair may
   * overwrite the latest value so that not all values written will be
   * read.
   * 
   * The core design is based on brilliantsugar (2024
   * @cite brilliantsugar2024triplebuffer), How I learned to stop worrying
   * and love juggling C++ atomics.
   * 
   * @tparam T Type of object buffered.
   */  
  template <typename T>
  class SpscBuffer {
  public:

    /**
     * @brief Construct a buffer filled with copies of the specified value.
     *
     * @param[in] t Value to write.
     */
    explicit SpscBuffer(const T& t) : buffers_{{t}, {t}, {t}} {}

    /**
     * @brief Construct a buffer filled with default-constructed values.
     */    
    SpscBuffer() requires std::default_initializable<T> : SpscBuffer(T()) {}

    SpscBuffer(const SpscBuffer&) = delete;
    SpscBuffer& operator=(const SpscBuffer&) = delete;
    SpscBuffer(SpscBuffer&&) = delete;
    SpscBuffer& operator=(SpscBuffer&&) = delete;

    /**
     * @brief Return a reference to the latest value.
     *
     * @return A refernece to the lateest value.
     */
    const T& read_latest() noexcept {
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
    T& write_buffer() noexcept { return buffers_[back_].data; }

    /**
     * @brief Commit the latest state of the write buffer.
     */
    void publish() noexcept {
      std::uint32_t prev = middle_.exchange(make_dirty(back_), std::memory_order_acq_rel);
      back_ = index_of(prev);
    }

  private:
    /** @brief Bit to mark an index as dirty */
    static constexpr std::uint32_t DIRTY_BIT = 0x80000000u;  

    /** @brief Mask to return the index. */
    static constexpr std::uint32_t INDEX_MASK = ~DIRTY_BIT;

    /**
     * @brief Return the dirty form of the index.
     *
     * @see index_of to recover the underlying index.
     *
     * @param[in] idx Index to make dirty.
     * @return The dirty form of the index.
     */
    static std::uint32_t make_dirty(std::uint32_t idx) noexcept { return idx | DIRTY_BIT; }

    /** 
     * @brief Return the clean form of the specified index.
     *
     * @param[in] A potentially dirty index.
     * @return The index underlying the argument in {0, 1, 2}.
     */
    static std::uint32_t index_of(std::uint32_t idx) noexcept { return idx & INDEX_MASK; }

    /**
     * @brief Return true if the index is dirty.
     *
     * @param idx The index.
     * @return `true` if the index is dirty.
     */
    static bool is_dirty(std::uint32_t idx) noexcept { return (idx & DIRTY_BIT) != 0; }

    /**
     * @brief A structure to hold an aligned form of the type `T`.
     */
    struct alignas(FALSE_SHARING_GUARD_SIZE) AlignedT { T data{}; };

    AlignedT buffers_[3];
    std::atomic<std::uint32_t> middle_{1};
    alignas(FALSE_SHARING_GUARD_SIZE) std::uint32_t back_{0};
    alignas(FALSE_SHARING_GUARD_SIZE) std::uint32_t front_{2};
  };

}
}
