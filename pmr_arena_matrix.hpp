#ifndef PMR_ARENA_MATRIX_HPP
#define PMR_ARENA_MATRIX_HPP

#include <Eigen/Dense>
#include <memory_resource>
#include <type_traits>

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

namespace nuts {

/**
 * A wrapper around Eigen::Map that uses a std::pmr::polymorphic_allocator
 * for memory allocation.
 *
 * @tparam MatrixType Eigen matrix type this works as (e.g., MatrixXd, VectorXd).
 */
template <typename MatrixType>
class pmr_arena_matrix : public Eigen::Map<std::decay_t<MatrixType>> {
 public:
  using Scalar = typename std::decay_t<MatrixType>::Scalar;
  using Base = Eigen::Map<std::decay_t<MatrixType>>;
  using allocator_t = std::pmr::polymorphic_allocator<Scalar>;
  using NestedExpression = typename Eigen::internal::remove_all<Base>::type;
  static constexpr int RowsAtCompileTime = MatrixType::RowsAtCompileTime;
  static constexpr int ColsAtCompileTime = MatrixType::ColsAtCompileTime;
  allocator_t allocator_;
  bool owns_data_{true};
  /**
   * Constructs a pmr_arena_matrix with the given allocator and dimensions.
   * @param allocator The polymorphic allocator to use for memory allocation.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  pmr_arena_matrix(allocator_t allocator, Eigen::Index rows, Eigen::Index cols)
      : Base(allocator.allocate(rows * cols), rows, cols), allocator_(allocator), owns_data_(true) {}

  /**
   * Constructs a pmr_arena_matrix with the given allocator and size.
   * This constructor is for row or column vectors.
   * @param allocator The polymorphic allocator to use for memory allocation.
   * @param size Number of elements.
   */
  pmr_arena_matrix(allocator_t allocator, Eigen::Index size)
      : Base(allocator.allocate(size), size), allocator_(allocator), owns_data_(true) {}

  /**
   * Constructs a pmr_arena_matrix from an Eigen expression.
   * @param allocator The polymorphic allocator to use for memory allocation.
   * @param other The Eigen expression to initialize the matrix with.
   */
  template <typename Expr, require_eigen<Expr>* = nullptr>
  pmr_arena_matrix(allocator_t allocator, const Expr& other)
      : Base::Map(
            allocator.allocate(other.size()),
            (RowsAtCompileTime == 1 && Expr::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && Expr::RowsAtCompileTime == 1)
                ? other.cols()
                : other.rows(),
            (RowsAtCompileTime == 1 && Expr::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && Expr::RowsAtCompileTime == 1)
                ? other.rows()
                : other.cols()),
        allocator_(allocator), owns_data_(true) {
    Base::operator=(other);
  }

  template <typename Expr, require_eigen<Expr>* = nullptr>
  explicit pmr_arena_matrix(const pmr_arena_matrix<Expr>& other)
      : Base::Map(
            other.allocator_.allocate(other.size()),
            (RowsAtCompileTime == 1 && Expr::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && Expr::RowsAtCompileTime == 1)
                ? other.cols()
                : other.rows(),
            (RowsAtCompileTime == 1 && Expr::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && Expr::RowsAtCompileTime == 1)
                ? other.rows()
                : other.cols()),
        allocator_(other.allocator_), owns_data_(true) {
    Base::operator=(other);
  }
  template <typename Expr, require_eigen<Expr>* = nullptr>
  explicit pmr_arena_matrix(pmr_arena_matrix<Expr>&& other)
      : Base::Map(
            const_cast<Scalar*>(other.data()),
            (RowsAtCompileTime == 1 && Expr::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && Expr::RowsAtCompileTime == 1)
                ? other.cols()
                : other.rows(),
            (RowsAtCompileTime == 1 && Expr::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && Expr::RowsAtCompileTime == 1)
                ? other.rows()
                : other.cols()),
        allocator_(other.allocator_), owns_data_(other.owns_data_) {
    other.owns_data_ = false;
  }
  explicit pmr_arena_matrix(const pmr_arena_matrix<MatrixType>& other)
      : Base::Map(
            const_cast<allocator_t&>(other.allocator_).allocate(other.size()),
            (RowsAtCompileTime == 1 && MatrixType::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && MatrixType::RowsAtCompileTime == 1)
                ? other.cols()
                : other.rows(),
            (RowsAtCompileTime == 1 && MatrixType::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && MatrixType::RowsAtCompileTime == 1)
                ? other.rows()
                : other.cols()),
        allocator_(other.allocator_), owns_data_(true) {
    Base::operator=(other);
  }
  explicit pmr_arena_matrix(pmr_arena_matrix<MatrixType>&& other)
      : Base::Map(
            const_cast<Scalar*>(other.data()),
            (RowsAtCompileTime == 1 && MatrixType::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && MatrixType::RowsAtCompileTime == 1)
                ? other.cols()
                : other.rows(),
            (RowsAtCompileTime == 1 && MatrixType::ColsAtCompileTime == 1)
                    || (ColsAtCompileTime == 1 && MatrixType::RowsAtCompileTime == 1)
                ? other.rows()
                : other.cols()),
        allocator_(other.allocator_), owns_data_(other.owns_data_) {
        other.owns_data_ = false;
  }

  /**
   * Destructor to deallocate memory.
   */
  ~pmr_arena_matrix() { if (owns_data_) allocator_.deallocate(this->data(), this->size()); }

  using Base::operator=;
  pmr_arena_matrix& operator=(const pmr_arena_matrix<MatrixType>& other) {
    Base::operator=(other);
    return *this;
  }
  pmr_arena_matrix& operator=(pmr_arena_matrix<MatrixType>&& other) {
    new (this) pmr_arena_matrix(std::move(other));
    return *this;
  }

  /**
   * Copy assignment operator for Eigen expressions.
   * @param other The Eigen expression to assign.
   * @return Reference to this matrix.
   */
  template <typename Expr>
  pmr_arena_matrix& operator=(const Expr& other) {
    Base::operator=(other);
    return *this;
  }

};

}  // namespace nuts

namespace Eigen {
namespace internal {

template <typename T>
struct traits<nuts::pmr_arena_matrix<T>> : traits<Eigen::Map<T>> {
  using base = traits<Eigen::Map<T>>;
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
