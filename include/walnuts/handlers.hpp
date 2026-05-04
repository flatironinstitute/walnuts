#pragma once

#include <iostream>

#include <Eigen/Dense>

#include "walnuts/validate.hpp"

namespace walnuts {

/**
 * @brief A handler that stores global events.
 */
class GlobalStore {
public: 
  /**
   * @brief Handle the R-hat value.
   *
   * @param r_hat The R-hat value.
   */
  void on_r_hat(double r_hat) { r_hats_.push_back(r_hat); }

  /**
   * @brief Return the R-hat values.
   *
   * @return The R-hat values.
   */
  const std::vector<double>& r_hats() { return r_hats_; }

private: 
  std::vector<double> r_hats_;
};


/**
 * @brief A handler that stores chain-local events.
 */
class ChainStore {
 public:
  /**
   * @brief Construct a chain store that optionally saves warmup iterations.
   *
   * @param[in] save_warmup Set to `true` to save warmup iterations.
   */
  ChainStore(bool save_warmup = false) : save_warmup_(save_warmup) {}

  /**
   * @brief Handle a warmup draw with meta-information.
   *
   * If `save_warmup()` is `true`, this operation appends the values
   * to vectors for later use.  Otherwise, it does nothing if
   * `save_warmup()` is `false`.
   * 
   * @param[in] position The position observed.
   * @param[in] lp The log density of the position observed.
   * @param[in] step_size The step size observed.
   * @param[in] diag_inv_mass The diagonal of the inverse mass matrix observed.
   */
  void on_warmup(const Eigen::VectorXd& position, const double lp,
                 const double step_size, const Eigen::VectorXd& diag_inv_mass) {
    if (!save_warmup_) {
      return;
    }
    warmup_draws_.push_back(position);
    warmup_lps_.push_back(lp);
    warmup_stepsizes_.push_back(step_size);
    warmup_diag_inv_masses_.push_back(diag_inv_mass);
  }

  /**
   * @brief Handle the warmup completion event.
   * 
   * @param step_size The step size.
   * @param diag_inv_mass The diagonal inverse mass matrix.
   */
  void on_warmup_complete(double step_size,
                          const Eigen::VectorXd& diag_inv_mass) {
    stepsize_ = step_size;
    diag_inv_mass_ = diag_inv_mass;
  }

  /**
   * @brief Handle a sampling event.
   *
   * @param position The position.
   * @param lp The log density.
   */
  void on_sample(const Eigen::VectorXd& position, double lp) {
    draws_.push_back(position);
    lps_.push_back(lp);
  }

  /**
   * @brief Handle an event to stop sampling.
   */
  void on_stop() {
    // TODO: add stop semaphore, catch interrupts
  }

  /**
   * @brief Return `true` if warmup iterations are saved.
   *
   * @return `true` if warmup iterations are saved.
   */
  bool save_warmup() const noexcept {
    return save_warmup_;
  }

  /**
   * @brief Return the step size.
   *
   * This method only makes sense once `on_warmup_complete()` has been called.
   *
   * @return The step size.
   */
  double step_size() const noexcept {
    return stepsize_;
  }

  /**
   * @brief Return the diagonal inverse mass matrix.
   *
   * @return The inverse mass matrix.
   */
  const Eigen::VectorXd& diag_inv_mass() const noexcept {
    return diag_inv_mass_;
  }

  /**
   * @brief Return the draws from sampling.
   *
   * @return The sampling draws.
   */
  const std::vector<Eigen::VectorXd>& draws() const noexcept {
    return draws_;
  }

  /**
   * @brief Return the log densities for the draws from sampling.
   *
   * @return The sampling log densities.
   */
  const std::vector<double>& log_probs() const noexcept {
    return lps_;
  }

  /**
   * @brief Return the draws from warmup.
   *
   * @return The warmup draws.
   */
  const std::vector<Eigen::VectorXd>& warmup_draws() const noexcept {
    return draws_;
  }

  /**
   * @brief Return the log densities for the draws from warmup.
   *
   * @return The warmup log densities.
   */
  const std::vector<double>& warmup_log_probs() const noexcept {
    return lps_;
  }

  /**
   * @brief Return the step sizes from warmup.
   *
   * @return The warmup step sizes.
   */
  const std::vector<double>& warmup_step_sizes() const noexcept {
    return warmup_stepsizes_;
  }

  /**
   * @brief Return the diagonal inverse mass matrices from warmup.
   *
   * @return The warmup mass matrices.
   */
  const std::vector<Eigen::VectorXd>& warmup_diag_inv_masses() const noexcept {
    return warmup_diag_inv_masses_;
  }

private:
  bool save_warmup_;

  double stepsize_ = 0;
  Eigen::VectorXd diag_inv_mass_;

  std::vector<double> r_hats_;

  std::vector<Eigen::VectorXd> draws_;
  std::vector<double> lps_;

  std::vector<Eigen::VectorXd> warmup_draws_;
  std::vector<double> warmup_lps_;
  std::vector<double> warmup_stepsizes_;
  std::vector<Eigen::VectorXd> warmup_diag_inv_masses_;
};

/**
 * @brief Write the step sizes to comma-separated-value format.
 *
 * @param[out] handlers The storage handlers for the chains.
 * @param[out] os The output stream.
 * @param[in] precision The number of significant digits in scalar output.
 */
inline void write_step_size_csv(const std::vector<ChainStore>& handlers,
				std::ostream& os, int precision = 8) {
  if (handlers.empty()) {
    return;
  }
  os << std::setprecision(precision);
  os << "chain_id,stepsize\n";
  for (std::size_t c = 0; c < handlers.size(); ++c) {
    os << c << ',' << handlers[c].step_size() << '\n';
  }
}

/**
 * @brief Write the diagonal mass matrices to comma-separated-value format.
 *
 * @param[out] handlers The storage handlers for the chains.
 * @param[out] os The output stream.
 * @param[in] precision The number of significant digits in scalar output.
 */
inline void write_mass_matrix_csv(const std::vector<ChainStore>& handlers,
				  std::ostream& os, int precision = 8) {
    if (handlers.empty()) {
      return;
    }
    std::int64_t D = handlers[0].diag_inv_mass().size();
    if (D == 0) {
      return;
    }
    os << std::setprecision(precision);
    os << "chain_id";
    for (std::int64_t d = 0; d < D; ++d) {
      os << ",theta[" << d << "]";
    }
    os << '\n';
    for (std::size_t c = 0; c < handlers.size(); ++c) {
      os << c;
      for (int d = 0; d < D; ++d) {
        os << ',' << handlers[c].diag_inv_mass()(d);
      }
      os << '\n';
    }
  }

/**
 * @brief Write the draws to comma-separated-value format, optionally
 * including warmup draws.
 *
 * @param[out] handlers The storage handlers for the chains.
 * @param[out] os The output stream.
 * @param[in] include_warmup `true` if warmup draws are included in output.
 * @param[in] precision The number of significant digits in scalar output.
 */
inline void write_sample_csv(const std::vector<ChainStore>& handlers,
			     std::ostream& os, bool include_warmup = false,
			     int precision = 8) {
  os << std::setprecision(precision);

  // get dims D from first draw
  int64_t D = 0;
  for (const auto& h : handlers) {
    if (!h.draws().empty()) {
      D = h.draws()[0].size();
      break;
    }
    if (!h.warmup_draws().empty()) {
      D = h.warmup_draws()[0].size();
      break;
    }
  }
  if (D == 0) {
    return;
  }

  os << "chain_id,warmup,iteration,log_density";
  for (int d = 0; d < D; ++d) {
    os << ",theta[" << d << "]";
  }
  os << '\n';

  for (std::size_t c = 0; c < handlers.size(); ++c) {
    const auto& h = handlers[c];
    int iteration = 0;

    if (include_warmup) {
      for (std::size_t n = 0; n < h.warmup_draws().size(); ++n, ++iteration) {
	os << c << ",1," << iteration << ',' << h.warmup_log_probs()[n];
	for (int d = 0; d < D; ++d) {
	  os << ',' << h.warmup_draws()[n](d);
	}
	os << '\n';
      }
    }

    for (std::size_t n = 0; n < h.draws().size(); ++n, ++iteration) {
      os << c << ",0," << iteration << ',' << h.log_probs()[n];
      for (int d = 0; d < D; ++d) {
	os << ',' << h.draws()[n](d);
      }
      os << '\n';
    }
  }
}

/**
 * @brief Write the step sizes to comma-separated-value format.
 *
 * @param[out] handlers The storage handlers for the chains.
 * @param[out] file_name The name of the file to which output is written.
 * @param[in] precision The number of significant digits in scalar output.
 * @throw std::invalid_argument If the file cannot be opened for writing.
 */  
inline void write_step_size_csv(const std::vector<ChainStore>& handlers,
				const std::string& file_name,
				int precision = 8) {
  std::ofstream os(file_name);
  walnuts::validate_open(os, file_name);
  write_step_size_csv(handlers, os, precision);
}

/**
 * @brief Write the diagonal mass matrices to comma-separated-value format.
 *
 * @param[out] handlers The storage handlers for the chains.
 * @param[out] file_name The name of the file to which output is written.
 * @param[in] precision The number of significant digits in scalar output.
 * @throw std::invalid_argument If the file cannot be opened for writing.
 */  
inline void write_mass_matrix_csv(const std::vector<ChainStore>& handlers,
				  const std::string& file_name,
				  int precision = 8) {
  std::ofstream os(file_name);
  walnuts::validate_open(os, file_name);
  write_mass_matrix_csv(handlers, os, precision);
}

/**
 * @brief Write the draws to comma-separated-value format, optionally
 * including warmup draws.
 *
 * @param[out] handlers The storage handlers for the chains.
 * @param[out] file_name The name of the file to which output is written.
 * @param[in] include_warmup `true` if warmup draws are included in output.
 * @param[in] precision The number of significant digits in scalar output.
 * @throw std::invalid_argument If the file cannot be opened for writing.
 */
inline void write_sample_csv(const std::vector<ChainStore>& handlers,
			     const std::string& file_name,
			     bool include_warmup = false, int precision = 8) {
  std::ofstream os(file_name);
  walnuts::validate_open(os, file_name);
  write_sample_csv(handlers, os, include_warmup, precision);
}  

}
