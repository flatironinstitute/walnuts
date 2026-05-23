#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

#include <walnuts.hpp>

// *** CLASS:  MarkovChainsSplit 

//   chain 0: [[ 1,  2], [ 3,  4]]
//   chain 1: [[ 5,  6], [ 7,  8], [ 9, 10]]
//   chain 2: [[11, 12], [13, 14], [15, 16]]
std::vector<Eigen::MatrixXd> make_example_chains() {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c0(2, 2);
  c0 << 1, 2,
        3, 4;
  Eigen::MatrixXd c1(3, 2);
  c1 << 5,  6,
        7,  8,
        9, 10;
  Eigen::MatrixXd c2(3, 2);
  c2 << 11, 12,
    13, 14,
    15, 16;
  chains.push_back(std::move(c0));
  chains.push_back(std::move(c1));
  chains.push_back(std::move(c2));
  return chains;
}

// constructor

TEST(MarkovChainsSplit, ConstructorThrowsOnEmptyChains) {
  std::vector<Eigen::MatrixXd> chains;
  EXPECT_THROW(walnuts::MarkovChainsSplit{chains}, std::invalid_argument);
}

TEST(MarkovChainsSplit, ConstructorThrowsOnZeroRowChain) {
  std::vector<Eigen::MatrixXd> chains;
  chains.emplace_back(Eigen::MatrixXd::Zero(2, 3));
  chains.emplace_back(Eigen::MatrixXd(0, 3));  // 0-row chain
  EXPECT_THROW(walnuts::MarkovChainsSplit{chains}, std::invalid_argument);
}

TEST(MarkovChainsSplit, ConstructorThrowsOnInconsistentNumberOfColumns) {
  std::vector<Eigen::MatrixXd> chains;
  chains.emplace_back(Eigen::MatrixXd::Zero(2, 3));
  chains.emplace_back(Eigen::MatrixXd::Zero(2, 4));  // mismatched cols
  EXPECT_THROW(walnuts::MarkovChainsSplit{chains}, std::invalid_argument);
}

// accessors

TEST(MarkovChainsSplit, SizeAccessorsReturnExpectedValues) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  EXPECT_EQ(mcs.num_chains(), std::size_t{3});
  EXPECT_EQ(mcs.num_draws(), std::size_t{8});
  EXPECT_EQ(mcs.dims(), std::size_t{2});
  EXPECT_EQ(mcs.min_chain_size(), Eigen::Index{2});
}

// views

TEST(MarkovChainsSplit, ChainViewReturnsCorrectChain) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  for (std::size_t m = 0; m < chains.size(); ++m) {
    Eigen::MatrixXd view = mcs.chain_view(m);
    ASSERT_EQ(view.rows(), chains[m].rows());
    ASSERT_EQ(view.cols(), chains[m].cols());
    EXPECT_TRUE(view.isApprox(chains[m]));
  }
}

TEST(MarkovChainsSplit, ChainViewThrowsOnOutOfRangeIndex) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  EXPECT_THROW(mcs.chain_view(3), std::out_of_range);   // one past end
  EXPECT_THROW(mcs.chain_view(99), std::out_of_range);  // far past end
}

// ---- draws -----------------------------------------------------------------

TEST(MarkovChainsSplit, DrawsConcatenatesAcrossChains) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);

  Eigen::VectorXd expected_d0(8);
  expected_d0 << 1, 3, 5, 7, 9, 11, 13, 15;
  Eigen::VectorXd expected_d1(8);
  expected_d1 << 2, 4, 6, 8, 10, 12, 14, 16;

  Eigen::VectorXd d0 = mcs.draws(0);
  Eigen::VectorXd d1 = mcs.draws(1);

  ASSERT_EQ(d0.size(), Eigen::Index{8});
  ASSERT_EQ(d1.size(), Eigen::Index{8});
  EXPECT_TRUE(d0.isApprox(expected_d0));
  EXPECT_TRUE(d1.isApprox(expected_d1));
}

TEST(MarkovChainsSplit, DrawsThrowsOnOutOfRangeDimension) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  EXPECT_THROW(mcs.draws(-1), std::out_of_range);
  EXPECT_THROW(mcs.draws(2), std::out_of_range);
  EXPECT_THROW(mcs.draws(100), std::out_of_range);
}

// ---- Degenerate-but-valid edge case ----------------------------------------

TEST(MarkovChainsSplit, SingleChainSingleDrawSingleDimension) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(1, 1);
  c << 42.0;
  chains.push_back(std::move(c));

  walnuts::MarkovChainsSplit mcs(chains);
  EXPECT_EQ(mcs.num_chains(), std::size_t{1});
  EXPECT_EQ(mcs.num_draws(), std::size_t{1});
  EXPECT_EQ(mcs.dims(), std::size_t{1});
  EXPECT_EQ(mcs.min_chain_size(), Eigen::Index{1});

  Eigen::VectorXd d = mcs.draws(0);
  ASSERT_EQ(d.size(), Eigen::Index{1});
  EXPECT_DOUBLE_EQ(d(0), 42.0);
}
