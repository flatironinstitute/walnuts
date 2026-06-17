#ifndef WALNUTPIE_INTERRUPT_HPP
#define WALNUTPIE_INTERRUPT_HPP

#include <csignal>
#include <cstring>

#include "errors.hpp"

#if WALNUTPIE_ON_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace walnutpie {
namespace interrupt {

static volatile std::sig_atomic_t interrupted = false;

/**
 * @brief Interrupt handler for Stan
 *
 * Wrapper around OS-specific interrupt handling. This class is used to
 * interrupt Stan's algorithms when the user presses `Ctrl+C`.
 *
 * This uses RAII to install a custom signal handler which is later
 * removed, restoring the previous handler if one existed.
 */
class walnutpie_interrupt_handler {
#if !WALNUTPIE_ON_WINDOWS  // POSIX signals
 public:
  walnutpie_interrupt_handler() {
    interrupted = false;

    memset(&custom, 0, sizeof(custom));
    sigemptyset(&custom.sa_mask);
    sigaddset(&custom.sa_mask, SIGINT);
    custom.sa_flags = SA_RESETHAND;
    custom.sa_handler = &walnutpie_interrupt_handler::signal_handler;
    sigaction(SIGINT, &custom, &before);
  }

  /**
   * Restore the original signal handler. Important for languages like Python's
   * REPL where `Ctrl+C` is used to interrupt the current command, not terminate
   * the program.
   */
  virtual ~walnutpie_interrupt_handler() { sigaction(SIGINT, &before, NULL); }

  static void signal_handler(int signal) { interrupted = true; }

 private:
  struct sigaction before;
  struct sigaction custom;

#else  // Windows
 public:
  walnutpie_interrupt_handler() {
    interrupted = false;

    SetConsoleCtrlHandler(walnutpie_interrupt_handler::signal_handler, TRUE);
  }

  /**
   * Remove our custom signal handler. Important for languages like Python's
   * REPL where `Ctrl+C` is used to interrupt the current command, not terminate
   * the program.
   */
  virtual ~walnutpie_interrupt_handler() {
    SetConsoleCtrlHandler(walnutpie_interrupt_handler::signal_handler, FALSE);
  }

  static BOOL WINAPI signal_handler(DWORD type) {
    switch (type) {
      case CTRL_C_EVENT:
      case CTRL_BREAK_EVENT:
        interrupted = true;
        return TRUE;
      default:
        return FALSE;
    }
  }
#endif

 public:
  void throw_if_interrupted() const {
    if (interrupted) {
      throw walnutpie::error::interrupt_exception();
    }
  }

  walnutpie_interrupt_handler(const walnutpie_interrupt_handler&) = delete;
  walnutpie_interrupt_handler(walnutpie_interrupt_handler&&) = delete;
  walnutpie_interrupt_handler operator=(const walnutpie_interrupt_handler&) =
      delete;
  walnutpie_interrupt_handler operator=(walnutpie_interrupt_handler&&) = delete;
};

}  // namespace interrupt
}  // namespace walnutpie
#endif
