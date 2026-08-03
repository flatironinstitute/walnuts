#ifndef WALNUTPY_ERRORS_HPP
#define WALNUTPY_ERRORS_HPP

#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

typedef enum {
  generic = 0,   ///< A generic runtime error from Stan.
  config = 1,    ///< An invalid configuration for the algorithm.
  interrupt = 2  ///< The user interrupted the algorithm with `Ctrl+C`.
} WalnutpyErrorType;

struct WalnutpyError {
 public:
  WalnutpyError(const char* msg,
                WalnutpyErrorType type = WalnutpyErrorType::generic)
      : msg(msg), type(type) {}

  std::string msg;
  WalnutpyErrorType type;
};

namespace walnutpy {
namespace error {

/**
 * Exception thrown when the user interrupts the program.
 * See walnutpie::interrupt::walnutpie_interrupt_handler for more details.
 */
class interrupt_exception {};

/**
 * Catches exceptions and stores them in a WalnutpyError.
 *
 * This returns the result of the function if it succeeds.
 * If it fails, it returns -1 if the function returns an int,
 * nullptr if the function returns a pointer, and void otherwise.
 */
template <typename F>
inline auto catch_exceptions(WalnutpyError** err, F f) {
  try {
    return f();
  } catch (const interrupt_exception& e) {
    if (err != nullptr) {
      *err = new WalnutpyError("", WalnutpyErrorType::interrupt);
    }
  } catch (const std::invalid_argument& e) {
    if (err != nullptr) {
      *err = new WalnutpyError(e.what(), WalnutpyErrorType::config);
    }
  } catch (const std::exception& e) {
    if (err != nullptr) {
      *err = new WalnutpyError(e.what());
    }
  } catch (...) {
    if (err != nullptr) {
      *err = new WalnutpyError("Unknown error");
    }
  }

  using Result = std::invoke_result_t<F>;
  if constexpr (std::is_same_v<Result, int>) {
    return -1;
  } else if constexpr (std::is_pointer_v<Result>) {
    return static_cast<Result>(nullptr);
  } else {
    static_assert(std::is_same_v<Result, void>, "Unexpected return type");
  }
}

template <typename T>
inline void check_nonnegative(const char* name, T val) {
  if (val < 0) {
    std::stringstream msg;
    msg << name << " must be non-negative, was " << val;
    throw std::invalid_argument(msg.str());
  }
}

}  // namespace error
}  // namespace walnutpy

#endif
