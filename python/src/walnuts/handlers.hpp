#include <cstddef>
#include <iomanip>
#include <sstream>

#include <Eigen/Dense>
#include <walnuts/load_stan.hpp>

namespace walnutpie {
using walnuts::DynamicStanModel;
using walnuts::unique_bs_rng;

/**
 * Callback used for printing.
 */
typedef void (*PRINT_CALLBACK)(const char* msg, size_t len, bool bad);

class PythonPrinter {
 public:
  PythonPrinter(const PRINT_CALLBACK print, std::size_t chain_id,
                std::size_t refresh)
      : print_(print), id_(chain_id), refresh_(refresh) {}

  bool should_output() const { return refresh_ != 0; }

  void print(const std::string& s) const {
    print_(s.c_str(), s.length(), false);
  }

  void print_exception(const std::exception& exn) const noexcept {
    std::stringstream ss;
    ss << "Chain [" << id_
       << "]:Error evaluating the log density during iteration " << iter_ + 1
       << ": " << exn.what() << std::endl;
    print(ss.str());
  }

  void end_warmup() { in_warmup_ = false; }
  void progress_report() {
    iter_++;
    if (!should_output() || iter_ % refresh_ != 0) {
      return;
    }

    std::stringstream ss;
    ss << "Chain [" << id_ << "]: Iteration " << iter_ << "\t"
       << (in_warmup_ ? "(Warmup)" : "(Sampling)") << std::endl;
    print(ss.str());
  }

 private:
  void print_stderr(const std::string& s) const {
    print_(s.c_str(), s.length(), true);
  }

  const PRINT_CALLBACK print_;
  const std::size_t id_, refresh_;
  std::size_t iter_ = 0;
  bool in_warmup_ = true;
};

class StanBufferHandler;

class BufferHandler {
 public:
  BufferHandler(double* out, double* stepsize_out, double* inv_metric_out,
                bool save_warmup, PythonPrinter& printer)
      : save_warmup_(save_warmup),
        out(out),
        stepsize_out(stepsize_out),
        inv_metric_out(inv_metric_out),
        p(printer) {}

  void on_sample(const Eigen::VectorXd& position, double lp) {
    Eigen::Map<Eigen::VectorXd>(out + written_ * position.size(),
                                position.size()) = position;
    written_++;
    p.progress_report();
  }

  void on_warmup(const Eigen::VectorXd& position, double lp, double step_size,
                 const Eigen::VectorXd& diag_inv_mass) {
    if (save_warmup_) {
      Eigen::Map<Eigen::VectorXd>(out + written_ * position.size(),
                                  position.size()) = position;
      written_++;
      written_warmup_++;
    }
    p.progress_report();
  }

  void on_warmup_complete(double step_size, const Eigen::VectorXd& inv_metric) {
    p.end_warmup();
    if (stepsize_out != nullptr) {
      *stepsize_out = step_size;
    }
    if (inv_metric_out != nullptr) {
      std::copy(inv_metric.data(), inv_metric.data() + inv_metric.size(),
                inv_metric_out);
    }
  }

  void on_logp_exception(const Eigen::VectorXd& pos,
                         const std::exception& e) noexcept {
    p.print_exception(e);
  }

  int written_sampling() const { return written_ - written_warmup_; }
  int written_warmup() const { return written_warmup_; }

 private:
  PythonPrinter& p;
  const bool save_warmup_;
  std::size_t written_ = 0, written_warmup_ = 0;
  double *out, *stepsize_out, *inv_metric_out;
  friend StanBufferHandler;
};

class StanBufferHandler : public BufferHandler {
 public:
  StanBufferHandler(const DynamicStanModel& model, unique_bs_rng& rng,
                    double* out, double* stepsize_out, double* inv_metric_out,
                    bool save_warmup, PythonPrinter& printer)
      : BufferHandler(out, stepsize_out, inv_metric_out, save_warmup, printer),
        model_(model),
        rng_(rng) {}

  void on_sample(const Eigen::VectorXd& position, double lp) {
    constrain(position);
    p.progress_report();
  }

  void on_warmup(const Eigen::VectorXd& position, double lp, double step_size,
                 const Eigen::VectorXd& diag_inv_mass) {
    if (save_warmup_) {
      constrain(position);
      written_warmup_++;
    }
    p.progress_report();
  }

 private:
  void constrain(const Eigen::VectorXd& position) {
    auto output = Eigen::Map<Eigen::VectorXd>(
        out + written_ * model_.constrained_dimensions(),
        model_.constrained_dimensions());
    try {
      Eigen::VectorXd params;
      model_.constrain_draw(position, output, rng_);
    } catch (const std::exception& exn) {
      p.print_exception(exn);
      output.array() = std::numeric_limits<double>::quiet_NaN();
    }
    written_++;
  }

  const DynamicStanModel& model_;
  unique_bs_rng& rng_;
};

class GlobalHandler {
 public:
  GlobalHandler(const PythonPrinter& printer) : p(printer) {}

  void on_r_hat(double rhat) {
    if (!p.should_output()) {
      return;
    }

    std::stringstream ss;
    ss << "Controller: R-hat at " << std::setprecision(10) << rhat << std::endl;
    p.print(ss.str());
  }

 private:
  const PythonPrinter& p;
};

}  // namespace walnutpie
