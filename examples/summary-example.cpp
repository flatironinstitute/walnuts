#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include <walnuts/summary.hpp>

int main() {
  Eigen::Index num_draws = 128;
  Eigen::Index dims = 4;
  
  Eigen::VectorXd lp(num_draws);
  Eigen::MatrixXd draws(num_draws, dims);
  draws.setRandom();
  for (Eigen::Index d = 0; d < dims; ++d) {
    draws.col(d).array() *= (d + 1.0) * (d + 1.0); // * sigma
    draws.col(d).array() += (d + 2.0) * (d + 2.0); // + mu
  }
    
  std::vector<std::size_t> chain_id{
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1,
      2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2
      };
      
  
  walnuts::MarkovChains chains(lp, draws, chain_id);

  std::cout << "num_chains = " << chains.num_chains() << "\n";
  std::cout << "num_draws = " << chains.num_draws() << "\n";
  std::cout << "dims = " << chains.dims() << "\n";

  std::cout << "chain_ids = ";
  for (std::size_t i = 0; i < chains.chain_id().size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << chains.chain_id()[i];
  }
  std::cout << "\n";

  std::cout << "draws:\n";
  for (Eigen::Index m = 0; m < num_draws; ++m) {
    std::cout << draws.row(m) << "\n";
  }
  std::cout << "\n";

  std::cout << "mean: " << walnuts::mean(chains) << "\n\n";

  std::cout << "sample variance: " << walnuts::sample_variance(chains) << "\n\n";
  
  std::cout << "sample standard_deviation: " << walnuts::sample_standard_deviation(chains) << "\n\n";

  Eigen::VectorXd probs(3);
  probs << 0.05, 0.5, 0.95;
  Eigen::MatrixXd quantiles = walnuts::quantiles(chains, probs);
  std::cout << "quantiles(" << probs.transpose() << "):\n" << quantiles << "\n\n";

  Eigen::RowVectorXd r_hat = walnuts::r_hat(chains);
  std::cout << "R-hat = " << r_hat << "\n\n";
  
}
