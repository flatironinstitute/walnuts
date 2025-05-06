#ifndef INCLUDE_NUTS_MEMORY_ARENA_ALLOC_HPP
#define INCLUDE_NUTS_MEMORY_ARENA_ALLOC_HPP


#include <Eigen/Dense>
#include <stdint.h>
#include <cstdlib>
#include <cstddef>
#include <vector>
#include "macros.hpp"



namespace nuts {

namespace internal {
/**
 * Return <code>true</code> if the specified pointer is aligned
 * on the number of bytes.
 *
 * This doesn't really make sense other than for powers of 2.
 *
 * @tparam Alignment Number of bytes of alignment required.
 * @tparam T Type of object to which pointer points.
 * @param ptr Pointer to test.
 * @return <code>true</code> if pointer is aligned.
 * @tparam Type of object to which pointer points.
 */
template <unsigned int Alignment, typename T>
NUTS_ALWAYS_INLINE bool is_aligned(T* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) % Alignment) == 0U;
}

constexpr size_t DEFAULT_INITIAL_NBYTES = 1 << 16;  // 64KB

/**
 * Allocates a block of memory of the specified size, returning
 * a pointer to the block if successful.
 * @param size Number of bytes to allocate.
 */
NUTS_ALWAYS_INLINE unsigned char* eight_byte_aligned_malloc(size_t size) {
  unsigned char* ptr = static_cast<unsigned char*>(malloc(size));
  if (!ptr) {
    return ptr;  // malloc failed to alloc
  }
  if (!is_aligned<8U>(ptr)) {
    [](auto* ptr) NUTS_COLD_PATH {
      std::stringstream s;
      s << "invalid alignment to 8 bytes, ptr="
        << reinterpret_cast<uintptr_t>(ptr) << std::endl;
      throw std::runtime_error(s.str());
    }(ptr);
  }
  return ptr;
}

}  // namespace internal

/**
 * An instance of this class provides a memory pool through
 * which blocks of raw memory may be allocated and then collected
 * simultaneously.
 *
 * This class is useful in settings where large numbers of small
 * objects are allocated and then collected all at once.  This may
 * include objects whose destructors have no effect.
 *
 * Memory is allocated on a stack of blocks.  Each block allocated
 * is twice as large as the previous one.  The memory may be
 * recovered, with the blocks being reused, or all blocks may be
 * freed, resetting the stack of blocks to its original state.
 *
 * Alignment up to 8 byte boundaries guaranteed for the first malloc,
 * and after that it's up to the caller.  On 64-bit architectures,
 * all struct values should be padded to 8-byte boundaries if they
 * contain an 8-byte member or a virtual function.
 */
class arena_alloc {
 public:
  using byte_t = unsigned char;
  std::vector<byte_t*> blocks_;  // storage for blocks,
                                 // may be bigger than cur_block_
  std::vector<size_t> sizes_;    // could store initial & shift for others
  size_t cur_block_;             // index into blocks_ for next alloc
  byte_t* cur_block_end_;        // ptr to cur_block_ptr_ + sizes_[cur_block_]
  byte_t* next_loc_;             // ptr to next available spot in cur
                                 // block
  // next three for keeping track of nested allocations on top of stack:
  std::vector<size_t> nested_cur_blocks_;
  std::vector<byte_t*> nested_next_locs_;
  std::vector<byte_t*> nested_cur_block_ends_;
  /**
   * Moves us to the next block of memory, allocating that block
   * if necessary, and allocates len bytes of memory within that
   * block.
   *
   * @param len Number of bytes to allocate.
   * @return A pointer to the allocated memory.
   */
  NUTS_NO_INLINE byte_t* move_to_next_block(size_t len) {
    byte_t* result;
    ++cur_block_;
    // Find the next block (if any) containing at least len bytes.
    while ((cur_block_ < blocks_.size()) && (sizes_[cur_block_] < len)) {
      ++cur_block_;
    }
    // Allocate a new block if necessary.
    if (unlikely(cur_block_ >= blocks_.size())) {
      // New block should be max(2*size of last block, len) bytes.
      size_t newsize = sizes_.back() * 2;
      if (newsize < len) {
        newsize = len;
      }
      blocks_.push_back(internal::eight_byte_aligned_malloc(newsize));
      if (!blocks_.back()) {
        throw std::bad_alloc();
      }
      sizes_.push_back(newsize);
    }
    result = blocks_[cur_block_];
    // Get the object's state back in order.
    next_loc_ = result + len;
    cur_block_end_ = result + sizes_[cur_block_];
    return result;
  }

 public:
  /**
   * Construct a resizable stack allocator initially holding the
   * specified number of bytes.
   *
   * @param initial_nbytes Initial number of bytes for the
   * allocator.  Defaults to <code>(1 << 16) = 64KB</code> initial bytes.
   * @throws std::runtime_error if the underlying malloc is not 8-byte
   * aligned.
   */
  NUTS_NO_INLINE explicit arena_alloc(size_t initial_nbytes)
      : blocks_(1, internal::eight_byte_aligned_malloc(initial_nbytes)),
        sizes_(1, initial_nbytes),
        cur_block_(0),
        cur_block_end_(blocks_[0] + initial_nbytes),
        next_loc_(blocks_[0]) {
    if (unlikely(!blocks_[0])) {
      throw std::bad_alloc();  // no msg allowed in bad_alloc ctor
    }
  }
  NUTS_NO_INLINE arena_alloc()
      : blocks_(1, internal::eight_byte_aligned_malloc(
                       internal::DEFAULT_INITIAL_NBYTES)),
        sizes_(1, internal::DEFAULT_INITIAL_NBYTES),
        cur_block_(0),
        cur_block_end_(blocks_[0] + internal::DEFAULT_INITIAL_NBYTES),
        next_loc_(blocks_[0]) {
    if (unlikely(!blocks_[0])) {
      throw std::bad_alloc();  // no msg allowed in bad_alloc ctor
    }
  }
  arena_alloc(const arena_alloc&) = delete;
  arena_alloc& operator=(const arena_alloc&) = delete;
  arena_alloc(arena_alloc&&) = delete;
  arena_alloc& operator=(arena_alloc&&) = delete;

  /**
   * Destroy this memory allocator.
   *
   * This is implemented as a no-op as there is no destruction
   * required.
   */
  NUTS_NO_INLINE ~arena_alloc() {
    // free ALL blocks
    for (auto& block : blocks_) {
      if (block) {
        free(block);
      }
    }
  }

  /**
   * Return a newly allocated block of memory of the appropriate
   * size managed by the stack allocator.
   *
   * The allocated pointer will be 8-byte aligned. If the number
   * of bytes requested is not a multiple of 8, the reserved space
   * will be padded up to the next multiple of 8.
   *
   * This function may call C++'s <code>malloc()</code> function,
   * with any exceptions percolated through this function.
   *
   * @param len Number of bytes to allocate.
   * @return A pointer to the allocated memory.
   */
  NUTS_ALWAYS_INLINE void* alloc(size_t len) {
    if (unlikely(len == size_t{0})) {
      return nullptr;
    }
    size_t pad = len % 8 == 0 ? 0 : 8 - len % 8;

    // Typically, just return and increment the next location.
    byte_t* result = next_loc_;
    next_loc_ += len + pad;
    // Occasionally, we have to switch blocks.
    if (unlikely(next_loc_ >= cur_block_end_)) {
      result = move_to_next_block(len);
    }
    return reinterpret_cast<void*>(result);
  }

  /**
   * Allocate an array on the arena of the specified size to hold
   * values of the specified template parameter type.
   *
   * @tparam T type of entries in allocated array.
   * @param[in] n size of array to allocate.
   * @return new array allocated on the arena.
   */
  template <typename T>
  NUTS_ALWAYS_INLINE T* alloc_array(size_t n) {
    return static_cast<T*>(alloc(n * sizeof(T)));
  }

  /**
   * Recover all the memory used by the stack allocator.  The stack
   * of memory blocks allocated so far will be available for further
   * allocations. If more than one block exists, all memory is freed
   * and then one large allocation takes place to allow only a single
   * block of memory. To free memory back to the system, use the
   * function free_all().
   */
  NUTS_ALWAYS_INLINE void recover_all() {
    if (unlikely(blocks_.size() > 1)) {
      std::size_t sum = 0;
      for (size_t i = 1; i < blocks_.size(); ++i) {
        sum += sizes_[i];
        free(blocks_[i]);
      }
      blocks_.clear();
      sizes_.clear();
      blocks_.push_back(internal::eight_byte_aligned_malloc(sum));
      sizes_.push_back(sum);
    }
    cur_block_ = 0;
    next_loc_ = blocks_[0];
    cur_block_end_ = next_loc_ + sizes_[0];
  }

  /**
   * Free all memory used by the stack allocator other than the
   * initial block allocation back to the system.  Note:  the
   * destructor will free all memory.
   */
  inline void free_all() {
    // frees all BUT the first (index 0) block
    for (size_t i = 1; i < blocks_.size(); ++i) {
      if (blocks_[i]) {
        free(blocks_[i]);
      }
    }
    sizes_.resize(1);
    blocks_.resize(1);
    recover_all();
  }

  /**
   * Return number of bytes allocated to this instance by the heap.
   * This is not the same as the number of bytes allocated through
   * calls to memalloc_.  The latter number is not calculatable
   * because space is wasted at the end of blocks if the next
   * alloc request doesn't fit.  (Perhaps we could trim down to
   * what is actually used?)
   *
   * @return number of bytes allocated to this instance
   */
  inline size_t bytes_allocated() const {
    size_t sum = 0;
    for (size_t i = 0; i <= cur_block_; ++i) {
      sum += sizes_[i];
    }
    return sum;
  }

  /**
   * Indicates whether the memory in the pointer
   * is in the stack.
   *
   * @param[in] ptr memory location
   * @return true if the pointer is in the stack,
   *    false otherwise.
   */
  inline bool in_stack(const void* ptr) const {
    for (size_t i = 0; i < cur_block_; ++i) {
      if (ptr >= blocks_[i] && ptr < blocks_[i] + sizes_[i]) {
        return true;
      }
    }
    if (ptr >= blocks_[cur_block_] && ptr < next_loc_) {
      return true;
    }
    return false;
  }

  /**
   * Store current positions before doing nested operation so can
   * recover back to start.
   */
  inline void start_nested() {
    nested_cur_blocks_.push_back(cur_block_);
    nested_next_locs_.push_back(next_loc_);
    nested_cur_block_ends_.push_back(cur_block_end_);
  }
    /**
   * recover memory back to the last start_nested call.
   */
  inline void recover_nested() {
    if (unlikely(nested_cur_blocks_.empty())) {
      recover_all();
    }

    cur_block_ = nested_cur_blocks_.back();
    nested_cur_blocks_.pop_back();

    next_loc_ = nested_next_locs_.back();
    nested_next_locs_.pop_back();

    cur_block_end_ = nested_cur_block_ends_.back();
    nested_cur_block_ends_.pop_back();
  }
};

template <typename Allocator>
struct scoped_allocator {
  std::reference_wrapper<Allocator> alloc_;
  explicit scoped_allocator(Allocator& alloc) : alloc_(alloc) {
    alloc_.get().start_nested();
  }

  ~scoped_allocator() { alloc_.get().recover_nested(); }
};

/**
 * std library compatible allocator that uses AD stack.
 * @tparam T type of scalar
 *
 * @warning The type T needs to be either trivially destructible or the dynamic
allocations needs to be managed by the arena_allocator.
 * For example this works: @code{.cpp}
using my_matrix = std::vector<std::vector<double,
stan::math::arena_allocator<double>>,
stan::math::arena_allocator<std::vector<double,
stan::math::arena_allocator<double>>>>;@endcode
 *
 */
template <typename T, typename ArenaType>
struct arena_allocator {
  ArenaType* alloc_;
  bool owns_alloc_{false};
  using value_type = T;
  NUTS_NO_INLINE explicit arena_allocator(ArenaType* alloc,
                                             bool owns_alloc = false)
      : alloc_(alloc), owns_alloc_(owns_alloc) {}
  NUTS_NO_INLINE arena_allocator()
      : alloc_(new ArenaType{}), owns_alloc_(true) {}

  NUTS_NO_INLINE arena_allocator(const arena_allocator& rhs)
      : alloc_(rhs.alloc_), owns_alloc_(false){};
  template <typename U, typename UArena>
  NUTS_NO_INLINE arena_allocator(const arena_allocator<U, UArena>& rhs)
      : alloc_(rhs.alloc_), owns_alloc_(false) {}
  template <typename U, typename UArena>
  NUTS_NO_INLINE arena_allocator(arena_allocator&& rhs)
      : alloc_(rhs.alloc_), owns_alloc_(rhs.owns_alloc_ ? true : false) {
    rhs.alloc_ = nullptr;
    rhs.owns_alloc_ = false;
  }

  ~arena_allocator() {
    if (owns_alloc_) {
      delete alloc_;
    }
  }

  /**
   * Allocates space for `n` items of type `T`.
   *
   * @param n number of items to allocate space for
   * @return pointer to allocated space
   */
  template <typename T_ = T>
  NUTS_ALWAYS_INLINE T_* allocate(std::size_t n) {
    return alloc_->template alloc_array<T_>(n);
  }

  /**
   * Recovers memory
   */
  NUTS_ALWAYS_INLINE void recover_memory() noexcept {
    alloc_->recover_all();
  }

  /**
   * No-op. Memory is deallocated by calling `recover_memory()`.
   */
  void deallocate(T* /*p*/, std::size_t /*n*/) noexcept {}

  /**
   * Equality comparison operator.
   * @return true
   */
  constexpr bool operator==(const arena_allocator&) const noexcept {
    return true;
  }
  /**
   * Inequality comparison operator.
   * @return false
   */
  constexpr bool operator!=(const arena_allocator&) const noexcept {
    return false;
  }
  void start_nested() { alloc_->start_nested(); }
  void recover_nested() { alloc_->recover_nested(); }
};



}  // namespace nuts
#endif
