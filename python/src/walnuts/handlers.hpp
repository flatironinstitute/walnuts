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

class StanBufferHandler;

class BufferHandler {
 public:
  BufferHandler(double* out, double* stepsize_out, double* inv_metric_out,
                bool save_warmup, std::size_t id, std::size_t refresh,
                PRINT_CALLBACK print)
      : save_warmup_(save_warmup),
        id_(id),
        refresh_(refresh),
        out(out),
        stepsize_out(stepsize_out),
        inv_metric_out(inv_metric_out),
        print(print) {}

  void on_sample(const Eigen::VectorXd& position, double lp) {
    progress_report();
    Eigen::Map<Eigen::VectorXd>(out + written_ * position.size(),
                                position.size()) = position;
    written_++;
  }

  void on_warmup(const Eigen::VectorXd& position, double lp, double step_size,
                 const Eigen::VectorXd& diag_inv_mass) {
    progress_report();
    if (!save_warmup_) {
      return;
    }
    Eigen::Map<Eigen::VectorXd>(out + written_ * position.size(),
                                position.size()) = position;
    written_++;
    written_warmup_++;
  }

  void on_warmup_complete(double step_size, const Eigen::VectorXd& inv_metric) {
    in_warmup_ = false;
    if (stepsize_out != nullptr) {
      *stepsize_out = step_size;
    }
    if (inv_metric_out != nullptr) {
      std::copy(inv_metric.data(), inv_metric.data() + inv_metric.size(),
                inv_metric_out);
    }
  }

  int written_sampling() const { return written_ - written_warmup_; }
  int written_warmup() const { return written_warmup_; }

  void on_logp_exception(const Eigen::VectorXd& pos,
                         const std::exception& e) noexcept {
    std::stringstream ss;
    ss << "Chain [" << id_
       << "]:Error evaluating the log density during iteration " << iter_ + 1
       << ": " << e.what() << std::endl;
    print_stderr(ss.str());
  }

 private:
  void progress_report() {
    iter_++;
    if (refresh_ == 0 || iter_ % refresh_ != 0) {
      return;
    }

    std::stringstream ss;
    ss << "Chain [" << id_ << "]: Iteration " << iter_ << "\t"
       << (in_warmup_ ? "(Warmup)" : "(Sampling)") << std::endl;
    print_stdout(ss.str());
  }

  void print_stdout(const std::string& s) {
    print(s.c_str(), s.length(), false);
  }

  void print_stderr(const std::string& s) {
    print(s.c_str(), s.length(), true);
  }

  PRINT_CALLBACK print;
  bool save_warmup_;
  bool in_warmup_ = true;
  std::size_t refresh_, id_;
  std::size_t iter_ = 0, written_ = 0, written_warmup_ = 0;
  double *out, *stepsize_out, *inv_metric_out;
  friend StanBufferHandler;
};

class StanBufferHandler : public BufferHandler {
 public:
  StanBufferHandler(const DynamicStanModel& model, unique_bs_rng& rng,
                    double* out, double* stepsize_out, double* inv_metric_out,
                    bool save_warmup, std::size_t id, std::size_t refresh,
                    PRINT_CALLBACK print)
      : BufferHandler(out, stepsize_out, inv_metric_out, save_warmup, id,
                      refresh, print),
        model_(model),
        rng_(rng) {}

  void on_sample(const Eigen::VectorXd& position, double lp) {
    progress_report();
    constrain(position);
  }

  void on_warmup(const Eigen::VectorXd& position, double lp, double step_size,
                 const Eigen::VectorXd& diag_inv_mass) {
    progress_report();
    if (!save_warmup_) {
      return;
    }
    constrain(position);
    written_warmup_++;
  }

 private:
  void constrain(const Eigen::VectorXd& position) {
    auto output = Eigen::Map<Eigen::VectorXd>(
        out + written_ * model_.constrained_dimensions(),
        model_.constrained_dimensions());
    try {
      Eigen::VectorXd params;
      model_.constrain_draw(position, output, rng_);
    } catch (...) {
      // TODO report
      output.array() = std::numeric_limits<double>::quiet_NaN();
    }
    written_++;
  }

  const DynamicStanModel& model_;
  unique_bs_rng& rng_;
};

class GlobalHandler {
 public:
  GlobalHandler(bool output, PRINT_CALLBACK print)
      : output_(output), print(print) {}

  void on_r_hat(double rhat) {
    if (!output_) {
      return;
    }

    std::stringstream ss;
    ss << "Controller: R-hat at " << std::setprecision(10) << rhat << std::endl;
    print_stdout(ss.str());
  }

 private:
  void print_stdout(const std::string& s) {
    print(s.c_str(), s.length(), false);
  }

  bool output_;
  PRINT_CALLBACK print;
};
}  // namespace walnutpie
