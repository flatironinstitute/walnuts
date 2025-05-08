#ifndef PMR_ARENA_MATRIX_HPP
#define PMR_ARENA_MATRIX_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <memory>
#include <memory_resource>
#include <type_traits>
#include <vector>

// TODO -- move traits or remove
template <template <typename> class Base, typename Derived>
struct is_base_pointer_convertible {
  static std::false_type f(const void*);
  template <typename OtherDerived>
  static std::true_type f(const Base<OtherDerived>*);
  enum {
    value
    = decltype(f(std::declval<std::remove_reference_t<Derived>*>()))::value
  };
};

template <template <typename> class Base, typename Derived>
inline constexpr bool is_base_pointer_convertible_v
    = is_base_pointer_convertible<Base, Derived>::value;

template <typename T>
struct is_eigen
    : std::bool_constant<
          is_base_pointer_convertible_v<Eigen::EigenBase, std::decay_t<T>>> {};

template <typename T>
inline constexpr bool is_eigen_v = is_eigen<T>::value;

template <typename T>
using require_eigen = std::enable_if_t<is_eigen_v<std::decay_t<T>>>;
#ifndef NUTS_INLINE
#define NUTS_INLINE __attribute__((always_inline)) inline
#endif
namespace nuts {
class pool_memory_resource final : public std::pmr::memory_resource {
public:
    /// block_size: size of each allocation (must be constant across calls)
    /// alignment: alignment of each block
    /// upstream: resource to grab fresh memory from
    pool_memory_resource(std::size_t block_size,
                         std::size_t alignment,
                         std::pmr::memory_resource* upstream = std::pmr::get_default_resource())
      : block_size_(block_size)
      , alignment_(alignment)
      , upstream_(upstream)
    {
        free_list_.reserve(std::pow(2, 18));
        all_blocks_.reserve(std::pow(2, 18));
        assert((alignment_ & (alignment_ - 1)) == 0 && "alignment must be a power of two");
    }

    ~pool_memory_resource() {
        // return all blocks we ever allocated back to the upstream
        for (void* p : all_blocks_) {
            upstream_->deallocate(p, block_size_, alignment_);
        }
    }

protected:
    NUTS_INLINE void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        // we only support requests <= block_size_ and alignment <= alignment_
        assert(bytes <= block_size_);
        assert(alignment <= alignment_);

        if (!free_list_.empty()) {
            void* p = free_list_.back();
            free_list_.pop_back();
            return p;
        }

        // no free blocks: grab a fresh one
        void* p = upstream_->allocate(block_size_, alignment_);
        all_blocks_.push_back(p);
        return p;
    }

    NUTS_INLINE void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
        assert(bytes <= block_size_);
        assert(alignment <= alignment_);
        // simply put it back into the free list
        free_list_.push_back(p);
    }

    NUTS_INLINE bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }
public:
    std::size_t                        block_size_;
    std::size_t                        alignment_;
    std::pmr::memory_resource*        upstream_;
    std::vector<void*>                 free_list_;
    std::vector<void*>                 all_blocks_;
};

/**
 * A wrapper around Eigen::Map that uses a std::pmr::polymorphic_allocator
 * for memory allocation.
 *
 * @tparam MatrixType Eigen matrix type this works as (e.g., MatrixXd, VectorXd).
 */
template <typename MatrixType>
class pmr_arena_matrix : public Eigen::Map<std::decay_t<MatrixType>, Eigen::Aligned128> {
 public:
  using Scalar = typename std::decay_t<MatrixType>::Scalar;
  using Base = Eigen::Map<std::decay_t<MatrixType>, Eigen::Aligned128>;
  using allocator_t = std::pmr::polymorphic_allocator<Scalar>;
  using NestedExpression = typename Eigen::internal::remove_all<Base>::type;
  static constexpr int RowsAtCompileTime = MatrixType::RowsAtCompileTime;
  static constexpr int ColsAtCompileTime = MatrixType::ColsAtCompileTime;
  struct buffy {
   allocator_t allocator_;
   alignas(128) Scalar* buffer_;
   Eigen::Index size_;
   NUTS_INLINE buffy(allocator_t allocator, std::size_t size) :
     allocator_(allocator),
     buffer_(reinterpret_cast<Scalar*>(allocator_.allocate_bytes(sizeof(Scalar) * size, 128))),
     size_(size) {}
  NUTS_INLINE buffy(const buffy& other) :
    allocator_(other.allocator_),
    buffer_(reinterpret_cast<Scalar*>(allocator_.allocate_bytes(sizeof(Scalar) * other.size_, 128))),
    size_(other.size_) {
      std::memcpy(buffer_, other.buffer_, sizeof(Scalar) * size_);
    }
  NUTS_INLINE buffy(buffy&& other) :
   allocator_(std::move(other.allocator_)),
    buffer_(other.buffer_),
    size_(other.size_) {
      other.buffer_ = nullptr;
      other.size_ = 0;
    }
   NUTS_INLINE ~buffy() {
      if (buffer_) allocator_.deallocate_bytes(buffer_, sizeof(Scalar) * size_, 128);
    }
  };
  buffy buffer_;
  /**
   * Constructs a pmr_arena_matrix with the given allocator and dimensions.
   * @param allocator The polymorphic allocator to use for memory allocation.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  NUTS_INLINE pmr_arena_matrix(allocator_t allocator, Eigen::Index rows, Eigen::Index cols)
      : Base(nullptr, rows, cols),
      buffer_(allocator, rows * cols) {
       new (this) Base(buffer_.buffer_, rows, cols);
      }

  /**
   * Constructs a pmr_arena_matrix with the given allocator and size.
   * This constructor is for row or column vectors.
   * @param allocator The polymorphic allocator to use for memory allocation.
   * @param size Number of elements.
   */
  NUTS_INLINE pmr_arena_matrix(allocator_t allocator, Eigen::Index size)
      : Base(nullptr, size),
            buffer_(allocator, size) {
       new (this) Base(buffer_.buffer_, size);
      }

  /**
   * Constructs a pmr_arena_matrix from an Eigen expression.
   * @param allocator The polymorphic allocator to use for memory allocation.
   * @param other The Eigen expression to initialize the matrix with.
   */
  template <typename Expr, require_eigen<Expr>* = nullptr>
  NUTS_INLINE pmr_arena_matrix(allocator_t allocator, const Expr& other)
      : Base::Map(
            nullptr, other.rows(), other.cols()),
        buffer_(allocator, other.size()) {
    new (this) Base(
            buffer_.buffer_,
            (RowsAtCompileTime == 1 && Expr::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && Expr::RowsAtCompileTime == 1)
                ? other.cols()
                : other.rows(),
            (RowsAtCompileTime == 1 && Expr::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && Expr::RowsAtCompileTime == 1)
                ? other.rows()
                : other.cols());
    (*this).noalias() = other;
  }

  template <typename Expr, require_eigen<Expr>* = nullptr>
  NUTS_INLINE explicit pmr_arena_matrix(const pmr_arena_matrix<Expr>& other)
      : Base::Map(nullptr, other.rows(), other.cols()),
        buffer_(other.buffer_) {
        new (this) Base(
            buffer_.buffer_,
            (RowsAtCompileTime == 1 && Expr::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && Expr::RowsAtCompileTime == 1)
                ? other.cols()
                : other.rows(),
            (RowsAtCompileTime == 1 && Expr::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && Expr::RowsAtCompileTime == 1)
                ? other.rows()
                : other.cols());
  }
  template <typename Expr, require_eigen<Expr>* = nullptr>
  NUTS_INLINE explicit pmr_arena_matrix(pmr_arena_matrix<Expr>&& other) noexcept
      : Base::Map(other.buffer_.buffer_, (RowsAtCompileTime == 1 && Expr::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && Expr::RowsAtCompileTime == 1)
                ? other.cols()
                : other.rows(),
            (RowsAtCompileTime == 1 && Expr::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && Expr::RowsAtCompileTime == 1)
                ? other.rows()
                : other.cols()),
        buffer_(std::move(other.buffer_)) {}
  NUTS_INLINE explicit pmr_arena_matrix(const pmr_arena_matrix<MatrixType>& other)
      : Base::Map(nullptr, other.rows(), other.cols()),
        buffer_(other.buffer_) {
        new (this) Base(
            buffer_.buffer_,
            (RowsAtCompileTime == 1 && MatrixType::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && MatrixType::RowsAtCompileTime == 1)
                ? other.cols()
                : other.rows(),
            (RowsAtCompileTime == 1 && MatrixType::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && MatrixType::RowsAtCompileTime == 1)
                ? other.rows()
                : other.cols());
  }
  NUTS_INLINE explicit pmr_arena_matrix(pmr_arena_matrix<MatrixType>&& other) noexcept
      : Base::Map(other.buffer_.buffer_, other.rows(), other.cols()),
        buffer_(std::move(other.buffer_)) {}

  using Base::operator=;
  NUTS_INLINE pmr_arena_matrix& operator=(const pmr_arena_matrix<MatrixType>& other) {
    std::memcpy(buffer_.buffer_, other.buffer_.buffer_,
              sizeof(Scalar) * other.size());
    return *this;
  }
  NUTS_INLINE pmr_arena_matrix& operator=(pmr_arena_matrix<MatrixType>&& other) {
    new (this) pmr_arena_matrix(std::move(other));
    return *this;
  }

  /**
   * Copy assignment operator for Eigen expressions.
   * @param other The Eigen expression to assign.
   * @return Reference to this matrix.
   */
  template <typename Expr>
  NUTS_INLINE pmr_arena_matrix& operator=(Expr&& other) noexcept {
    Base::operator=(std::forward<Expr>(other));
    return *this;
  }

};

}  // namespace nuts

namespace Eigen {
namespace internal {

template <typename T>
struct traits<nuts::pmr_arena_matrix<T>> : traits<Eigen::Map<T, Eigen::Aligned128>> {
  using base = traits<Eigen::Map<T, Eigen::Aligned128>>;
  using XprKind = typename Eigen::internal::traits<std::decay_t<T>>::XprKind;
  using Scalar = typename std::decay_t<T>::Scalar;
  enum {
    PlainObjectTypeInnerSize = base::PlainObjectTypeInnerSize,
    InnerStrideAtCompileTime = base::InnerStrideAtCompileTime,
    OuterStrideAtCompileTime = base::OuterStrideAtCompileTime,
    Alignment = base::Alignment,
    Flags = base::Flags
  };
};
}
}

#endif
