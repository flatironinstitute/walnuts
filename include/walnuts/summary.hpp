#pragma once

#include <stdexcept>
#include <Eigen/Dense>

namespace walnuts {

  /**
   * @brief A class for representing a collection of Markov chains.
   * 
   * The class can be constructed from either sorted long-form data
   * with chain identifiers for each row or a collection of matrices,
   * one per chain.
   */
  class MarkovChains {
  public:
    MarkovChains(Eigen::VectorXd lps,
		 Eigen::MatrixXd draws,
		 std::vector<std::size_t> chain_id)
      : lps_(lps), draws_(draws), chain_id_(chain_id) {
      if (lps.rows() != draws.rows()) {
	throw std::invalid_argument("lps and draws must have same number of rows.");
      }
      if (lps.rows() != static_cast<Eigen::Index>(chain_id.size())) {
	throw std::invalid_argument("number of rows in lps must match size of chain_id");
      }
      if (chain_id.size() < 1) {
	throw std::invalid_argument("require at least one draw");
      }
      
      for (std::size_t n = 1; n < chain_id.size(); ++n) {
	if (chain_id[n] < chain_id[n-1]) {
	  throw std::invalid_argument("chain IDs must be in increasing order");
	}
	if (chain_id[n] != chain_id[n - 1]) {
	  chain_end_.push_back(n);
	}
      }
      chain_end_.push_back(chain_id.size());
    }

    std::size_t num_chains() const noexcept {
      return chain_end_.size();
    }

    std::size_t num_draws() const noexcept {
      return static_cast<std::size_t>(draws_.rows());
    }
    
    std::size_t dims() const noexcept {
      return static_cast<std::size_t>(draws_.cols());
    }

    const std::vector<std::size_t>& chain_id() const noexcept {
      return chain_id_;
    }

    const std::vector<std::size_t>& chain_ends() const noexcept {
      return chain_end_;
    }

    const Eigen::MatrixXd& draws() const noexcept {
      return draws_;
    }

  private:
    Eigen::VectorXd lps_;
    Eigen::MatrixXd draws_;
    std::vector<std::size_t> chain_id_;
    std::vector<std::size_t> chain_end_;  // one past end for ranges
  };

  template <typename Derived>
  static auto mean(const Eigen::MatrixBase<Derived>& draws) {
    return draws.colwise().mean().eval();
  }
  inline Eigen::RowVectorXd mean(const MarkovChains& chains) {
    return mean(chains.draws());
  }

  template <typename Derived>
  static auto sample_variance(const Eigen::MatrixBase<Derived>& draws,
			      const Eigen::RowVectorXd& mean) {
    return ((draws.rowwise() - mean).array().square().colwise().sum()
	    / (draws.rows() - 1)).eval();
  }
  template <typename Derived>
  static auto sample_variance(const Eigen::MatrixBase<Derived>& draws) {
    return ((draws.rowwise() - draws.colwise().mean()).array().square().colwise().sum()
	    / (draws.rows() - 1)).eval();
  }
  inline Eigen::RowVectorXd sample_variance(const MarkovChains& chains) {
    if (chains.num_draws() < 2) {
      throw std::domain_error("chains must have at least 2 draws");
    }
    return sample_variance(chains.draws());
  }

  template <typename Derived>
  static auto sample_standard_deviation(const Eigen::MatrixBase<Derived>& draws,
					const Eigen::RowVectorXd& mean) {
    return sample_variance(draws, mean).array().sqrt().matrix().eval();
  }
  template <typename Derived>
  static auto sample_standard_deviation(const Eigen::MatrixBase<Derived>& draws) {
    return sample_variance(draws).array().sqrt().matrix().eval();
  }
  inline Eigen::RowVectorXd sample_standard_deviation(const MarkovChains& chains) {
    if (chains.num_draws() < 2) {
      throw std::domain_error("chains must have at least 2 draws");
    }
    return sample_standard_deviation(chains.draws());
  }

  template <typename Derived>
  static auto population_variance(const Eigen::MatrixBase<Derived>& draws) {
    return ((draws.rowwise() - draws.colwise().mean()).array().square().colwise().sum()
	    / draws.rows()).eval();
  }
  inline Eigen::RowVectorXd population_variance(const MarkovChains& chains) {
    if (chains.num_draws() < 1) {
      throw std::invalid_argument("chains must have at least 1 draw");
    }
    return population_variance(chains.draws());
  }

  
  template <typename Derived>
  static auto population_standard_deviation(const Eigen::MatrixBase<Derived>& draws) {
    return population_variance(draws).array().sqrt().matrix().eval();
  }
  inline Eigen::RowVectorXd population_standard_deviation(const MarkovChains& chains) {
    if (chains.num_draws() < 1) {
      throw std::invalid_argument("chains must have at least 1 draw");
    }
    return population_standard_deviation(chains.draws());
  }

  template <typename Derived>
  static Eigen::MatrixXd quantiles(const Eigen::MatrixBase<Derived>& draws, const Eigen::VectorXd probs) {
    const Eigen::Index N = draws.rows();
    const Eigen::Index D = draws.cols();
    const Eigen::Index K = probs.size();
    Eigen::MatrixXd result(K, D);
    for (Eigen::Index d = 0; d < D; ++d) {
      Eigen::VectorXd col = draws.col(d);  // copy to not mutate draws
      std::sort(col.begin(), col.end());
      for (Eigen::Index k = 0; k < K; ++k) {
  	const double h = probs(k) * static_cast<double>(N - 1);
  	const Eigen::Index lo = static_cast<Eigen::Index>(std::floor(h));
  	const Eigen::Index hi = std::min(lo + 1, N - 1);  // lo + 1 == ceil(h)
  	const double frac = h - static_cast<double>(lo);
  	result(k, d) = col(lo) + frac * (col(hi) - col(lo));
      }
    }
    return result;
  }

  /**
   * @brief Return the empirical quantiles of the draws.
   *
   * The returned matrix has one quantile per row, with one dimension
   * per column.
   * 
   * For each variable, the empirical quantile at probability `p` is
   * computed by sorting the variable's column and linearly
   * interpolating between an upper bounding and lower bounding value.
   * This function's behavior matches R's `stats::quantile(x, probs,
   * type = 7)` (the default `type`) and NumPy's `numpy.quantile(a, q,
   * method='linear')` (the default `method`).
   *
   * In pseudocode, where `column` is the column of values and `p` is
   * the probability for the quantile function, the quantile is calculated
   * as follows.
   *
   * @code
   * sorted = sort(column)
   * idx = p * (N - 1)          // (N - 1) is last index
   * lb = floor(h)
   * ub = min(lb + 1, N - 1)    // don't go past last index
   * ub_frac = idx - lb         // distance toward upper bound
   * lb_frac = ub - idx         // lb_frac + ub_frac = 1
   * quantile = ub_frac * sorted[ub] + lb_frac * sorted[lb]
   * @endcode
   *
   * For example, if the values are `column = (9, 11, 5, 3)` and the 
   * probability is `p = 0.6`, we have 
   *
   * @code 
   * sorted                        = (3, 5, 9, 11)
   * idx = 0.6 * (4 - 1)           = 1.8
   * lb = floor(1.8)               = 1
   * ub = min(1 + 1, 4 - 1)        = 2
   * ub_frac = 1.8 - 1             = 0.8
   * lb_frac = 2 - 1.8             = 0.2
   * sorted[1]                     = 5
   * sorted[2]                     = 9
   * quantile = 0.8 * 9 + 0.2 * 5  = 8.2
   * @endcode
   *
   * @param[in] chains The Markov chains.
   * @param[in] probs A vector of probabilities in [0, 1].
   * @return The quantiles with one row per quantile.
   * @throw std::invalid_argument If `draws` has fewer than 2 draws.
   * @throw std::invalid_argument If a value in `probs` is outside [0, 1].
   */
  inline Eigen::MatrixXd quantiles(const MarkovChains& chains,
  				   const Eigen::VectorXd& probs) {
    if (chains.num_draws() < 2) {
      throw std::invalid_argument("chains must have at least two draws");
    }
    for (Eigen::Index k = 0; k < probs.size(); ++k) {
      if (!(probs(k) >= 0 && probs(k) <= 1.0)) {
  	throw std::invalid_argument("probs must be in [0, 1]");
      }
    }
    return quantiles(chains.draws(), probs);
  }

  static Eigen::RowVectorXd r_hat(const Eigen::MatrixXd& draws,
				  const std::vector<std::size_t>& chain_ends) {
    Eigen::Index K = static_cast<Eigen::Index>(chain_ends.size());
    Eigen::Index D = draws.cols();
    Eigen::MatrixXd mu(K, D);
    Eigen::MatrixXd sigma_sq(K, D);
    Eigen::Index start = 0;
    for (Eigen::Index k = 0; k < K; ++k) {
      Eigen::Index end = static_cast<Eigen::Index>(chain_ends[static_cast<std::size_t>(k)]);
      Eigen::Index length = end - start;
      auto chain = draws.middleRows(start, length);
      auto chain_mean = mean(chain);
      mu.row(k) = chain_mean;
      sigma_sq.row(k) = sample_standard_deviation(chain, chain_mean);
      start = end;
    }
    return (1.0 + sample_variance(mu).array() / mean(sigma_sq).array())
      .matrix();
  }

  /**
   * @brief Return the chain-balanced ragged R-hat statistic for the chains.
   *
   * The R-hat statistic weights the within-chain mean and variance of each
   * chain equally, no matter how long they are.  

   * The number of draws per chain may vary, so let `chain[k]` be the
   * `N[k] x D` matrix of draws for chain `k`. The means and variances
   * are taken column-wise as in the `walnuts::mean` and
   * `walnuts::sample_variance` functions. Sample variance divides by
   * `(N[k] - 1)` for an unbiased estimate of variance.
   * 
   * @code
   * matrix[K, D] mu, sigma_sq;
   * mu[k, ] = mean(chain[k])  for k in 1:K
   * sigma_sq[k, ] = sample_variance(chain[k])
   * R-hat = 1 + sample_variance(mu) ./ mean(sigma_sq)
   * @endcode
   *
   * See Gelman and Rubin (1992 @cite gelmanrubin1992) for the original definition of 
   * R-hat and Margossian (2025 @cite margossian2026) for the one used here.
   * 
   * @param[in] chains The Markov chains.
   * @return The R-hat statistic for each variable in the chain.
   */
  inline Eigen::RowVectorXd r_hat(const MarkovChains& chains) {
    if (chains.num_chains() < 2 || chains.num_draws() < 2) {
      throw std::invalid_argument("chains must have at least 2 chains and at least 2 draws");
    }
    return r_hat(chains.draws(), chains.chain_ends());
  }

  
  
} // namespace nuts
