#include <cstddef>

#include <Eigen/Dense>
#include <walnuts/load_stan.hpp>

namespace walnutpie {
using walnuts::DynamicStanModel;
using walnuts::unique_bs_rng;

class BufferHandler {
 public:
  BufferHandler(double* out, int num_params, double* stepsize_out,
                double* inv_metric_out, bool save_warmup)
      : num_params_(num_params),
        save_warmup_(save_warmup),
        n_(0),
        out(out),
        stepsize_out(stepsize_out),
        inv_metric_out(inv_metric_out) {}

  void on_sample(const Eigen::VectorXd& position, double lp) {
    Eigen::Map<Eigen::VectorXd>(out + n_ * num_params_, num_params_) = position;
    n_++;
  }

  void on_warmup(const Eigen::VectorXd& position, double lp, double step_size,
                 const Eigen::VectorXd& diag_inv_mass) {
    if (!save_warmup_) {
      return;
    }
    Eigen::Map<Eigen::VectorXd>(out + n_ * num_params_, num_params_) = position;
    n_++;
  }

  void on_warmup_complete(double step_size, const Eigen::VectorXd& inv_metric) {
    if (stepsize_out != nullptr) {
      *stepsize_out = step_size;
    }
    if (inv_metric_out != nullptr) {
      std::copy(inv_metric.data(), inv_metric.data() + inv_metric.size(),
                inv_metric_out);
    }
  }

  int written() const { return n_; }

 private:
  int num_params_;
  bool save_warmup_;
  std::size_t n_;
  double *out, *stepsize_out, *inv_metric_out;
};

class StanBufferHandler {
 public:
  StanBufferHandler(const DynamicStanModel& model, unique_bs_rng& rng,
                    double* out, double* stepsize_out, double* inv_metric_out,
                    bool save_warmup)
      : model_(model),
        rng_(rng),

        save_warmup_(save_warmup),
        n_(0),
        out(out),
        stepsize_out(stepsize_out),
        inv_metric_out(inv_metric_out) {}

  void on_sample(const Eigen::VectorXd& position, double lp) {
    constrain(position);
  }

  void on_warmup(const Eigen::VectorXd& position, double lp, double step_size,
                 const Eigen::VectorXd& diag_inv_mass) {
    if (!save_warmup_) {
      return;
    }
    constrain(position);
  }

  void on_warmup_complete(double step_size, const Eigen::VectorXd& inv_metric) {
    if (stepsize_out != nullptr) {
      *stepsize_out = step_size;
    }
    if (inv_metric_out != nullptr) {
      std::copy(inv_metric.data(), inv_metric.data() + inv_metric.size(),
                inv_metric_out);
    }
  }

  int written() const { return n_; }

 private:
  void constrain(auto&& in) {
    auto output =
        Eigen::Map<Eigen::VectorXd>(out + n_ * model_.constrained_dimensions(),
                                    model_.constrained_dimensions());
    try {
      Eigen::VectorXd params;
      model_.constrain_draw(in, output, rng_);
    } catch (...) {
      output.array() = std::numeric_limits<double>::quiet_NaN();
    }
    n_++;
  }

  const DynamicStanModel& model_;
  unique_bs_rng& rng_;
  bool save_warmup_;
  std::size_t n_;
  double *out, *stepsize_out, *inv_metric_out;
};

class DummyGlobalHandler {
 public:
  void on_r_hat(double) {}
};
}  // namespace walnutpie
