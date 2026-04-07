// include/walnuts/triple_buffer.hpp

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <utility>

#include "walnuts/padded.hpp"

namespace walnuts {

/*
 * @brief A lock-free, single-producer, single-consumer queue with triple buffering.
 *
 * @tparam T Type of object buffered.
 */
template <class T>
class TripleBuffer {
 public:

  /**
   * @brief Construct a buffer using the specified factory to
   * construct values of type `T` to buffer.  The factory must
   * be a functor that takes no arguments and returns an object
   * of type `T`.
   *
   * @tparam Factory The type of initial value generator.
   * @param[in] make The functor to create initial values.
   */
  template <class Factory>
  explicit TripleBuffer(Factory make)
      : buffers_{make(), make(), make()},
        front_(0),
        spare_(1),
        back_(2),
        read_(0) {}

  /**
   * Move the specified buffer into this buffer.
   *
   * @param other The buffer to move.
   */
  TripleBuffer(TripleBuffer&& other)
    noexcept(std::is_nothrow_move_constructible_v<std::array<T, 3>>)
    : buffers_(std::move(other.buffers_)),
      front_(other.front_.load(std::memory_order_relaxed)),
      spare_(other.spare_.load(std::memory_order_relaxed)),
      back_(other.back_),
      read_(other.read_) {}

  
  /**
   * Move the specified buffer into this buffer.
   *
   * @param other The buffer to move.
   * @return A reference to this buffer.
   */
  TripleBuffer& operator=(TripleBuffer&& other)
    noexcept(std::is_nothrow_move_assignable_v<std::array<T, 3>>) {
    if (this == &other) return *this;
    buffers_ = std::move(other.buffers_);
    front_.store(other.front_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    spare_.store(other.spare_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    back_ = other.back_;
    read_ = other.read_;
    return *this;
  }
  
  TripleBuffer(const TripleBuffer&) = delete;
  TripleBuffer& operator=(const TripleBuffer&) = delete;

  /**
   * Return the a reference into which to write a value.
   *
   * @return A reference into which to write a value.
   */
  T& write_buffer() noexcept { return buffers_[back_]; }

  /**
   * Commit the changes to the write buffer. 
   */
  void publish() noexcept {
    const int published = back_;
    front_.store(published, std::memory_order_release);
    back_ = spare_.exchange(published, std::memory_order_acq_rel);
  }

  /**
   * Return a constant reference to the latest value.
   *
   * @return The latest value.
   */
  const T& read_latest() noexcept {
    const int idx = front_.load(std::memory_order_acquire);
    if (idx != read_) {
      const int old = read_;
      read_ = idx;
      spare_.store(old, std::memory_order_release);
    }
    return buffers_[read_];
  }

 private:
  /** Type to use for indexing. */
  using index_t = int;

  /** Contiguous memory to store the buffer values.  */
  std::array<T, 3> buffers_;

  /** Current front of the buffer, aligned to avoid cache line conflicts. */
  alignas(walnuts::DI_SIZE) std::atomic<index_t> front_;

  /** Current spare buffer, aligned to avoid cache line conflicts. */
  alignas(walnuts::DI_SIZE) std::atomic<index_t> spare_;

  /** Current back buffer---only used by writer. */
  alignas(walnuts::DI_SIZE) index_t back_;

  /** Current read buffer---only used by reader. */
  alignas(walnuts::DI_SIZE) index_t read_;
};

}  // namespace walnuts  
