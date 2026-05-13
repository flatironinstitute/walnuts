#pragma once

#include <complex>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

namespace walnuts {

  namespace {
    template <typename Derived>
    static Eigen::RowVectorXd mean(const Eigen::MatrixBase<Derived>& draws) {
      return draws.colwise().mean().eval();
    }

    template <typename Derived>
    static Eigen::RowVectorXd sample_variance(const Eigen::MatrixBase<Derived>& draws,
                                              const Eigen::RowVectorXd& mean) {
      return ((draws.rowwise() - mean).array().square().colwise().sum()
              / (draws.rows() - 1));
    }

    template <typename Derived>
    static Eigen::RowVectorXd sample_variance(const Eigen::MatrixBase<Derived>& draws) {
      return ((draws.rowwise() - draws.colwise().mean()).array().square().colwise().sum()
              / (draws.rows() - 1));
    }

    // template <typename Derived>
    // static Eigen::MatrixXd quantiles(const Eigen::MatrixBase<Derived>& draws, const Eigen::VectorXd& probs) {
    //   const Eigen::Index N = draws.rows();
    //   const Eigen::Index D = draws.cols();
    //   const Eigen::Index K = probs.size();
    //   Eigen::MatrixXd result(K, D);
    //   for (Eigen::Index d = 0; d < D; ++d) {
    //     Eigen::VectorXd col = draws.col(d);  // copy to not mutate draws
    //     std::sort(col.begin(), col.end());
    //     for (Eigen::Index k = 0; k < K; ++k) {
    //       const double h = probs(k) * static_cast<double>(N - 1);
    //       const Eigen::Index lo = static_cast<Eigen::Index>(std::floor(h));
    //       const Eigen::Index hi = std::min(lo + 1, N - 1);  // lo + 1 == ceil(h)
    //       const double frac = h - static_cast<double>(lo);
    //       result(k, d) = col(lo) + frac * (col(hi) - col(lo));
    //     }
    //   }
    //   return result;
    // }


    static Eigen::RowVectorXd r_hat(const Eigen::MatrixXd& draws,
                                    const std::vector<Eigen::Index>& chain_starts,
				    const std::vector<Eigen::Index>& chain_sizes) {
      std::size_t K = chain_starts.size();
      Eigen::Index D = draws.cols();
      Eigen::MatrixXd mu(K, D);
      Eigen::MatrixXd sigma_sq(K, D);
      for (std::size_t k = 0; k < K; ++k) {
        auto chain = draws.middleRows(chain_starts[k], chain_sizes[k]);
        auto chain_mean = mean(chain);
        mu.row(static_cast<Eigen::Index>(k)) = mean(chain);
        sigma_sq.row(static_cast<Eigen::Index>(k)) = sample_variance(chain, chain_mean);
      }
      return (1.0 + sample_variance(mu).array() / mean(sigma_sq).array())
        .sqrt().matrix();
    }

    static void strip_factor(Eigen::Index& m, Eigen::Index factor) {
      while (m % factor == 0) {
        m /= factor;
      }
    }

    /**
     * @brief Return smallest number greater than or equal to 
     * input that is evenly divisible by 2, 3, and 5.
     *
     * @param n Original size.
     * @return Original size padded for divisibility.
     */
    static Eigen::Index fft_next_good_size(Eigen::Index n) {
      if (n <= 2) {
        return 2;
      }
      for (; true; ++n) {
        auto m = n;
        strip_factor(m, 2);
        strip_factor(m, 3);
        strip_factor(m, 5);
        if (m <= 1) {
          return n;
        }
      }
    }

    // implementation from stan-dev/stan/; BSD-3 licensed
    static void autocovariance_col(const Eigen::VectorXd& y,
                                   Eigen::VectorXd& ac,
                                   Eigen::FFT<double>& fft) {
      // TODO: evaluate the following optimization
      // fft.SetFlag(fft.HalfSpectrum);


      Eigen::Index N = y.size();
      Eigen::Index M = fft_next_good_size(N);
      Eigen::Index M2 = 2 * M;

      Eigen::VectorXd padded_signal(M2);
      padded_signal.setZero();
      padded_signal.head(N) = y.array() - y.mean();

      Eigen::VectorXcd freq_vec(M2);
      fft.fwd(freq_vec, padded_signal);
      freq_vec = freq_vec.cwiseAbs2();
      
      Eigen::VectorXcd ac_tmp(M2);
      fft.inv(ac_tmp, freq_vec);

      ac_tmp /= static_cast<double>(N);  // biased estimate recommended by Geyer (1992)
      ac = ac_tmp.head(N).real().array();
    }
    
    static Eigen::MatrixXd autocovariance_chain(const Eigen::MatrixXd& chain,
                                                Eigen::FFT<double>& fft) {
      Eigen::Index N = chain.rows();
      Eigen::Index D = chain.cols();
      Eigen::MatrixXd acor(N, D);
      Eigen::VectorXd acor_col(N);
      for (Eigen::Index d = 0; d < D; ++d) {
        Eigen::VectorXd col = chain.col(d);
        autocovariance_col(col, acor_col, fft);
        acor.col(d) = acor_col;
      }
      return acor;
    }

    // returns long form
    static Eigen::MatrixXd autocovariance(const Eigen::MatrixXd& draws,
                                          const std::vector<Eigen::Index>& chain_starts,
    					  const std::vector<Eigen::Index>& chain_sizes) {
      Eigen::FFT<double> fft;
      Eigen::Index N = draws.rows();
      Eigen::Index D = draws.cols();
      std::size_t M = chain_starts.size();
      Eigen::MatrixXd acov(N, D);
      for (std::size_t m = 0; m < M; ++m) {
        Eigen::MatrixXd chain = draws.middleRows(chain_starts[m], chain_sizes[m]);
        acov.middleRows(chain_starts[m], chain_sizes[m]) = autocovariance_chain(chain, fft);
      }
      return acov;
    }

    static Eigen::MatrixXd autocorrelation(const Eigen::MatrixXd& draws,
                                           const std::vector<Eigen::Index>& chain_starts,
					   const std::vector<Eigen::Index>& chain_sizes) {
      // acorr is not correlation until division
      Eigen::MatrixXd acorr = autocovariance(draws, chain_starts, chain_sizes);
      for (std::size_t m = 0; m < chain_starts.size(); ++m) {
        Eigen::RowVectorXd vars = acorr.row(chain_starts[m]);  // variance is autocovar at lag 0
        acorr.middleRows(chain_starts[m], chain_sizes[m]).array().rowwise() /= vars.array();
      }
      return acorr;
    }    
    
    static Eigen::Index min_size(std::vector<Eigen::Index> x) {
      Eigen::Index m = std::numeric_limits<Eigen::Index>::max();
      for (std::size_t k = 0; k < x.size(); ++k) {
        m = std::min(m, x[k]);
      }
      return m;
    }
    
    static Eigen::RowVectorXd ess(const Eigen::MatrixXd& draws,
                                  const std::vector<Eigen::Index>& chain_starts,
				  const std::vector<Eigen::Index>& chain_sizes) {
      Eigen::Index D = draws.cols();
      std::size_t K = chain_starts.size();
      Eigen::Index N_total = draws.rows();

      // chain starts and lengths
      Eigen::Index min_len = min_size(chain_sizes);

      // per-chain means and variances
      Eigen::MatrixXd chain_means(K, D);
      Eigen::MatrixXd chain_vars(K, D);
      for (std::size_t k = 0; k < K; ++k) {
        auto chain = draws.middleRows(chain_starts[k], chain_sizes[k]);
        Eigen::RowVectorXd m = mean(chain);
        chain_means.row(static_cast<Eigen::Index>(k)) = m;
        chain_vars.row(static_cast<Eigen::Index>(k)) = sample_variance(chain, m);
      }

      // W (within-chain variance) and var_plus (Margossian convention)
      Eigen::RowVectorXd W = mean(chain_vars);
      Eigen::RowVectorXd var_plus = W;
      if (K > 1) {
        var_plus += sample_variance(chain_means);
      }

      // autocovariance per chain, stacked ragged
      Eigen::MatrixXd acov = autocovariance(draws, chain_starts, chain_sizes);

      Eigen::RowVectorXd result(D);
      for (Eigen::Index d = 0; d < D; ++d) {
        const double w_d = W(d);
        const double vp_d = var_plus(d);

        // mean_acov_at_lag(t): average over chains of acov(chain_starts[k] + t, d)
        auto mean_acov_at_lag = [&](Eigen::Index t) {
          double sum = 0.0;
          for (std::size_t k = 0; k < K; ++k) {
            sum += acov(chain_starts[k] + t, d);
          }
          return sum / static_cast<double>(K);
        };

        Eigen::VectorXd rho_hat_t = Eigen::VectorXd::Zero(min_len);
        double rho_hat_even = 1.0;
        rho_hat_t(0) = rho_hat_even;

        double rho_hat_odd = 1.0 - (w_d - mean_acov_at_lag(1)) / vp_d;
        rho_hat_t(1) = rho_hat_odd;

        // Geyer's initial positive + monotone sequence on paired lags
        Eigen::Index t = 1;
        while (t < min_len - 4
               && (rho_hat_even + rho_hat_odd) > 0.0
               && !std::isnan(rho_hat_even + rho_hat_odd)) {
          rho_hat_even = 1.0 - (w_d - mean_acov_at_lag(t + 1)) / vp_d;
          rho_hat_odd  = 1.0 - (w_d - mean_acov_at_lag(t + 2)) / vp_d;

          if ((rho_hat_even + rho_hat_odd) >= 0.0) {
            rho_hat_t(t + 1) = rho_hat_even;
            rho_hat_t(t + 2) = rho_hat_odd;
          }
          // initial positive -> initial monotone
          if (rho_hat_t(t + 1) + rho_hat_t(t + 2)
              > rho_hat_t(t - 1) + rho_hat_t(t)) {
            rho_hat_t(t + 1) = (rho_hat_t(t - 1) + rho_hat_t(t)) / 2.0;
            rho_hat_t(t + 2) = rho_hat_t(t + 1);
          }
          t += 2;
        }

        Eigen::Index max_t = t;
        // antithetic-tail correction
        if (rho_hat_even > 0.0 && max_t + 1 < min_len) {
          rho_hat_t(max_t + 1) = rho_hat_even;
        }

        double tau_hat = -1.0
          + 2.0 * rho_hat_t.head(max_t).sum()
          + (max_t + 1 < min_len ? rho_hat_t(max_t + 1) : 0.0);

        // safety floor: tau >= 1/log10(N_total)
        tau_hat = std::max(tau_hat, 1.0 / std::log10(static_cast<double>(N_total)));

        result(d) = static_cast<double>(N_total) / tau_hat;
      }
      return result;
    }

    std::size_t sum(const std::vector<std::size_t>& vs) {
      std::size_t total = 0;
      for (auto v : vs) {
	total += v;
      }
      return total;
    }

    Eigen::MatrixXd concatenate_chains(const std::vector<Eigen::MatrixXd>& draws) {
      if (draws.size() == 0) return {};
      Eigen::Index N = 0;
      for (const Eigen::MatrixXd& chain : draws) {
	N += chain.rows();
      }
      Eigen::MatrixXd result(N, draws[0].cols());
      Eigen::Index start = 0;
      for (std::size_t m = 0; m < draws.size(); ++m) {
	result.middleRows(start, draws[m].rows()) = draws[m];
	start += draws[m].rows();
      }
      return result;
    }

    std::vector<std::size_t> chain_size(const std::vector<Eigen::MatrixXd> chains) {
      std::vector<std::size_t> sizes(chains.size());
      for (std::size_t m = 0; m < chains.size(); ++m) {
	sizes[m] = static_cast<std::size_t>(chains[m].rows());
      }
      return sizes;
    }
    
  }  // end namespace

  /**
   * @brief A class for representing a collection of Markov chains of
   * possibly varying lengths.
   */
  class MarkovChains {
  public:
    /**
     * @brief Construct an instance with the specified unnormalized
     * log densities, draws, and chain sizes.
     *
     * @param draws The sequence of Markov chains states.
     * @param chain_sizes The sizes of the Markov chains making up the
     * collection.
     * @throw std::invalid_argument If the sum of the chain sizes
     * isn't equal to the number of rows of the draws.
     */
    MarkovChains(const Eigen::MatrixXd& draws,
		 const std::vector<std::size_t>& chain_sizes)
      :	draws_(draws),
	chain_sizes_(chain_sizes.size()),
	chain_starts_(chain_sizes.size()),
	chain_ends_(chain_sizes.size()) {
      std::size_t total_size = sum(chain_sizes);
      if (total_size != static_cast<std::size_t>(draws.rows())) {
	throw std::invalid_argument("number of rows in draws and sum of chain_sizes must be equal.");
      }
      Eigen::Index total = 0;
      for (std::size_t m = 0; m < chain_sizes.size(); ++m) {
	chain_sizes_[m] = static_cast<Eigen::Index>(chain_sizes[m]);
	chain_starts_[m] = total;
	total += chain_sizes[m];
	chain_ends_[m] = total;
      }
    }

    /**
     * @brief Construct an instance with the specified log densities and draws per chain. 
     *
     * Each chain in the sequence of chains is organized with one draw per row, with
     * one column per variable to analyze.
     *
     * @param chains The sequence of Markov chains.
     * @throw std::invalid_argument If the chains do not all have the same number of columns.
     */
    MarkovChains(const std::vector<Eigen::MatrixXd>& chains):
      MarkovChains(concatenate_chains(chains),
		   chain_size(chains)) {
    }

    /**
     * @brief Return the number of Markov chains.
     *
     * @return The number of Markov chains.
     */
    std::size_t num_chains() const noexcept {
      return chain_ends_.size();
    }

    /**
     * @brief Return the total number of draws in all Markov chains.
     *
     * @return The total number of draws.
     */
    std::size_t num_draws() const noexcept {
      return static_cast<std::size_t>(draws_.rows());
    }
    
    /**
     * @brief Return the dimensionality of the draws.
     *
     * @return The dimensionality of the draws.
     */
    std::size_t dims() const noexcept {
      return static_cast<std::size_t>(draws_.cols());
    }

    /**
     * @brief Return the complete set of draws, with each
     * draw being a row.
     *
     * @return The draws for the Markov chains.
     */
    const Eigen::MatrixXd& draws() const noexcept {
      return draws_;
    }

    /**
     * @brief Return the sizes of the Markov chains.
     *
     * @return The Markov chain sizes.
     */
    const std::vector<Eigen::Index>& chain_sizes() const noexcept {
      return chain_sizes_;
    }

    /**
     * @brief Return the position of the first element in each chain.
     *
     * @return The positions where the chains start.
     */
    const std::vector<Eigen::Index>& chain_starts() const noexcept {
      return chain_starts_;
    }

    /**
     * @brief Return the position of one past the last element of each chain.
     *
     * @return The positions where the chains end plus one.
     */
    const std::vector<Eigen::Index>& chain_ends() const noexcept {
      return chain_ends_;
    }

    /**
     * @brief Return a modifiable view of the specified chain.
     *
     * @param n The index of the chain.
     * @return The specified chain.
     * @throw std::invalid_argument If the index is greater than or
     * equal to the number of chains.
     */
    auto chain_view(std::size_t n) {
      return draws_.middleRows(chain_starts_[n], chain_sizes_[n]);
    }

    auto const_chain_view(std::size_t n) const {
      return draws_.middleRows(chain_starts_[n], chain_sizes_[n]);
    }

    
    Eigen::VectorXd concatenate_columns(Eigen::Index d) const {
      return draws_.col(d);
    }
  private:
    Eigen::MatrixXd draws_;
    std::vector<Eigen::Index> chain_sizes_;  
    std::vector<Eigen::Index> chain_starts_;
    std::vector<Eigen::Index> chain_ends_;  // one past end
  };

  /**
   * @brief Return the sample means of the variables in the chains.
   *
   * The means are calculated for each variable (i.e., each
   * dimension). 
   * 
   * @param chains The Markov chains.
   * @return The sample means.
   */  
  inline Eigen::RowVectorXd mean(const MarkovChains& chains) {
    Eigen::RowVectorXd sum(chains.dims());
    for (std::size_t m = 0; m < chains.num_chains(); ++m) {
      sum += chains.const_chain_view(m).colwise().sum();
    }
    return sum / chains.num_draws();
  }

  /**
   * @brief Return the sample variances of the variables in the chains.
   *
   * The variances are calculated for each variable (i.e., each
   * dimension). The formula divides by the number of draws minus one
   * and thus provides an unbiased estimate of the population variance
   * based on a small sample.  If used to calculate the variance of an
   * entire population, it will be biased to the high side.
   * 
   * @param chains The Markov chains.
   * @return The variances.
   */  
  inline Eigen::RowVectorXd sample_variance(const MarkovChains& chains) {
    if (chains.num_draws() < 2) {
      throw std::domain_error("chains must have at least 2 draws");
    }
    Eigen::RowVectorXd mu = mean(chains);
    Eigen::RowVectorXd sum_sq(mu.size());
    for (std::size_t m = 0; m < chains.num_chains(); ++m) {
      auto chain = chains.const_chain_view(m);
      sum_sq = sum_sq + (chain.rowwise() - mu).array().square().colwise().sum().matrix();
    }
    return sum_sq / (chains.num_draws() - 1);
  }

  /**
   * @brief Return the sample standard deviations of the variables in the chains.
   *
   * The standard deviations are calculated for each variable (i.e., each
   * dimension). The formula divides by the number of draws minus one.  Nevertheless,
   * unlike the sample variance estimate, sample standard deviations are not unbiased
   * estimates of population standard deviations.
   * 
   * @param chains The Markov chains.
   * @return The standard deviations.
   */  
  inline Eigen::RowVectorXd sample_standard_deviation(const MarkovChains& chains) {
    return sample_variance(chains).array().sqrt().matrix();
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
   * @see <a href="https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html">R
   * `stats:quantile`function, `type=7`</a>

   * @see <a href="https://numpy.org/doc/stable/reference/generated/numpy.quantile.html">NumPy
   * (Python) `numpy.quantile` function,`method='linear'`</a> (Python).
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
    const Eigen::Index N = static_cast<Eigen::Index>(chains.num_draws());
    const Eigen::Index D = static_cast<Eigen::Index>(chains.dims());
    const Eigen::Index K = probs.size();
    Eigen::MatrixXd result(K, D);
    for (Eigen::Index d = 0; d < D; ++d) {
      Eigen::VectorXd col = chains.concatenate_columns(d);;  // copy to not mutate draws
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
   * @brief Return the matrix of autocovariances at all lags for each of the chains.
   *
   * The return will have the same shape as `chains.draws()`, with autocorrelation values
   * in place of parameter values.
   *
   * @param[in] The Markov chains.
   * @return The matrix of autocovariances.
   */
  Eigen::MatrixXd autocovariance(const MarkovChains& chains) {
    return autocovariance(chains.draws(), chains.chain_starts(), chains.chain_sizes());
    // std::cout << "STARTING" << std::endl;
    // Eigen::FFT<double> fft;
    // Eigen::Index N = static_cast<Eigen::Index>(chains.num_draws());
    // Eigen::Index D = static_cast<Eigen::Index>(chains.dims());
    // std::size_t M = chains.num_chains();
    // std::cout << "M = " << M << std::endl;							
    // Eigen::MatrixXd acov(N, D);
    // Eigen::Index start = 0;
    // for (std::size_t m = 0; m < M; ++m) {
    //   Eigen::MatrixXd chain = chains.const_chain_view(m);
    //   acov.middleRows(start, chain.rows()) = autocovariance_chain(chain, fft);
    // }
    // return acov;
  }
  
  /**
   * @brief Return the chain-balanced ragged R-hat statistics for the chains.
   *
   * The R-hat statistic weights the within-chain mean and variance of each
   * chain equally, no matter how long they are.  
   * 
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
   * See Gelman and Rubin (1992 @cite gelman1992inference) for the original definition of 
   * R-hat and Margossian (2025 @cite margossian2025nested) for the one used here.
   * 
   * @param[in] chains The Markov chains.
   * @return The R-hat statistic for each variable in the chain.
   */
  inline Eigen::RowVectorXd r_hat(const MarkovChains& chains) {
    if (chains.num_chains() < 2 || chains.num_draws() < 2) {
      throw std::invalid_argument("chains must have at least 2 chains and at least 2 draws");
    }
    return r_hat(chains.draws(), chains.chain_starts(), chains.chain_sizes());
  }

  /**
   * @brief Return the effective sample size statistics for the chains.
   *
   * The effective sample size is adjusted downward when R-hat is
   * greater than 1 (Gelman et al. 2013).  If only a single chain is
   * provided, there is no adjustment.
   *
   * The algorithm is \f$mathcal{O}(N log N)\f$ per dimension with N draws.  The
   * computational bottleneck is that autocovariances are calculated
   * with Eigen's built-in not-so-fast Fourier transform (FFT).
   *
   * The implementation is based on the one in Stan.  
   * 
   * @see Gelman et al. (2013 @cite gelman2013bda3) <a href="https://sites.stat.columbia.edu/gelman/book/"><i>Bayesian
   * Data Analysis</i></a>
   *
   * @see Margossian et al. (2005 @cite margossian2025nested) <a
   * href="https://projecteuclid.org/journals/bayesian-analysis/advance-publication/Nested-Rˆ--Assessing-the-Convergence-of-Markov-Chain-Monte/10.1214/24-BA1453.full">Nested
   * R-hat: Assessing the convergence of Markov chain Monte Carlo when
   * running many short chains</a>.
   * 
   * @see Stan Development Team. (2026 @cite standev2026ref). <a
   * href="https://mc-stan.org/docs/reference-manual/">Stan Reference
   * Manual</a>.
   * 
   * @see Stan Development Team. (2026).  C++ Source Code: <a
   * href="https://github.com/stan-dev/stan/blob/develop/src/stan/analyze/mcmc/ess.hpp">
   * Effective sample size</a> and <a
   * href="https://github.com/stan-dev/stan/blob/develop/src/stan/analyze/mcmc/rhat.hpp">R-hat</a>.
   * GitHub.
   * 
   * @param[in] chains The Markov chains.
   * @return The R-hat statistic for each variable in the chain.
   */
  inline Eigen::RowVectorXd effective_sample_size(const MarkovChains& chains) {
    if (chains.num_draws() < 3) {
      throw std::invalid_argument("chains must have at least 3 draws");
    }
    return ess(chains.draws(), chains.chain_starts(), chains.chain_sizes());
  }

  /**
   * @brief Return the Monte Carlo standard error for the chains.
   * 
   * The Monte Carlo standard error (MCSE) is just the sample standard deviation
   * divided by the square root of the effective sample size.
   *
   * @see `effective_sample_size(const MarkovChains&)`
   * 
   * @see `sample_standard_deviation(const MarkovChains&)`
   *
   * @param chains The Markov chains.
   * @return The per-chain Monte Carlo standard error estimates.
   */
  inline Eigen::RowVectorXd monte_carlo_standard_error(const MarkovChains& chains) {
    auto ess = effective_sample_size(chains);
    auto sd = sample_standard_deviation(chains);
    return (sd.array() / ess.array().sqrt()).matrix();
  }

}  // namespace walnuts

