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
 * @brief A mutex-protected single-producer, single-consumer buffer.
 *
 * @tparam T Type of object buffered.
 */
template <class T>
class alignas(FALSE_SHARING_GUARD_SIZE) SpscBuffer {
 public:
  /**
   * Construct a buffer with the specified value.
   *
   * @param[in] t Value to write.
   */
  explicit SpscBuffer(const T& t) : write_buf_(t), read_buf_(t) {}


  /**
   * @brief Construct a buffer with a default-constructed value.
   */
  SpscBuffer() : SpscBuffer(T()) {}

  /**
   * @brief Construct a triple buffer by moving the other buffer.
   *
   * @parma[in] other Buffer to move.
   */
  SpscBuffer(SpscBuffer&& other) noexcept(
      std::is_nothrow_move_constructible_v<T>)
      : write_buf_(std::move(other.write_buf_)),
        read_buf_(std::move(other.read_buf_)) {}

  /**
   * @brief Move the other buffer's contents into this buffer.
   *
   * @param other The buffer to move.
   * @return A refernece to this buffer.
   */
  SpscBuffer& operator=(SpscBuffer&& other) noexcept(
      std::is_nothrow_move_assignable_v<T>) {
    if (this == &other) {
      return *this;
    }
    write_buf_ = std::move(other.write_buf_);
    read_buf_ = std::move(other.read_buf_);
    return *this;
  }

  SpscBuffer(const SpscBuffer&) = delete;
  SpscBuffer& operator=(const SpscBuffer&) = delete;

  /**
   * Return a reference into which the writer stages a value.
   * Writer-only between publish() calls.
   *
   * @return The buffered value.
   */
  T& write_buffer() noexcept { return write_buf_; }

  /**
   * Commit the staged value: copies write_buf_ into read_buf_ under the lock.
   */
  void publish() noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    read_buf_ = write_buf_;
  }

  /**
   * Return a copy of the most recently published value.
   *
   * @return Most recently published value.
   */
  T read_latest() noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    return read_buf_;
  }

 private:
  /** @brief Buffer for writing. */
  T write_buf_;

  /** @brief Mutex used to guard reading. */
  alignas(FALSE_SHARING_GUARD_SIZE) std::mutex mtx_;

  /** @brief Buffer for reading. */
  T read_buf_;
};

}  // namespace detail
}  // namespace walnuts
