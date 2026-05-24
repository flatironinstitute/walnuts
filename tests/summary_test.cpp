#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <walnuts.hpp>

// MarkovChainsSplit class ====================================================

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

// MarkovChainsUnified class

// same chains as MarkovChainsSplit test
Eigen::MatrixXd make_unified_draws() {
  Eigen::MatrixXd draws(8, 2);
  draws << 1,  2,
           3,  4,
           5,  6,
           7,  8,
           9, 10,
          11, 12,
          13, 14,
          15, 16;
  return draws;
}

const std::vector<std::size_t> chain_sizes{2, 3, 3};

// constructor

TEST(MarkovChainsUnified, ConstructorThrowsOnChainSizesMismatch) {
  Eigen::MatrixXd draws = make_unified_draws();
  std::vector<std::size_t> wrong_sizes{2, 3, 2};
  EXPECT_THROW(walnuts::MarkovChainsUnified(draws, wrong_sizes),
               std::invalid_argument);
}

TEST(MarkovChainsUnified, ConstructorThrowsOnChainSizesTooLarge) {
  Eigen::MatrixXd draws = make_unified_draws();
  std::vector<std::size_t> wrong_sizes{2, 3, 4};
  EXPECT_THROW(walnuts::MarkovChainsUnified(draws, wrong_sizes),
               std::invalid_argument);
}

// accessors

TEST(MarkovChainsUnified, SizeAccessorsReturnExpectedValues) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnuts::MarkovChainsUnified mcu(draws, chain_sizes);
  EXPECT_EQ(mcu.num_chains(), std::size_t{3});
  EXPECT_EQ(mcu.num_draws(), std::size_t{8});
  EXPECT_EQ(mcu.dims(), std::size_t{2});
  EXPECT_EQ(mcu.min_chain_size(), Eigen::Index{2});
}

// chain view

TEST(MarkovChainsUnified, ChainViewReturnsCorrectRows) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnuts::MarkovChainsUnified mcu(draws, chain_sizes);

  // Expected chain contents match the logical split
  Eigen::MatrixXd expected0(2, 2);
  expected0 << 1, 2, 3, 4;
  Eigen::MatrixXd expected1(3, 2);
  expected1 << 5, 6, 7, 8, 9, 10;
  Eigen::MatrixXd expected2(3, 2);
  expected2 << 11, 12, 13, 14, 15, 16;

  EXPECT_TRUE(mcu.chain_view(0).isApprox(expected0));
  EXPECT_TRUE(mcu.chain_view(1).isApprox(expected1));
  EXPECT_TRUE(mcu.chain_view(2).isApprox(expected2));
}

TEST(MarkovChainsUnified, ChainViewIsAViewNotACopy) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnuts::MarkovChainsUnified mcu(draws, chain_sizes);
  Eigen::Ref<const Eigen::MatrixXd> view = mcu.chain_view(0);
  EXPECT_EQ(view.data(), draws.data());  // tests memory sharing
}

TEST(MarkovChainsUnified, ChainViewThrowsOnOutOfRangeIndex) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnuts::MarkovChainsUnified mcu(draws, chain_sizes);
  EXPECT_THROW(mcu.chain_view(3), std::out_of_range);
  EXPECT_THROW(mcu.chain_view(99), std::out_of_range);
}

// draws

TEST(MarkovChainsUnified, DrawsReturnsCorrectColumn) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnuts::MarkovChainsUnified mcu(draws, chain_sizes);

  Eigen::VectorXd expected_d0(8);
  expected_d0 << 1, 3, 5, 7, 9, 11, 13, 15;
  Eigen::VectorXd expected_d1(8);
  expected_d1 << 2, 4, 6, 8, 10, 12, 14, 16;

  ASSERT_EQ(mcu.draws(0).size(), Eigen::Index{8});
  ASSERT_EQ(mcu.draws(1).size(), Eigen::Index{8});
  EXPECT_TRUE(mcu.draws(0).isApprox(expected_d0));
  EXPECT_TRUE(mcu.draws(1).isApprox(expected_d1));
}

TEST(MarkovChainsUnified, DrawsIsAViewNotACopy) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnuts::MarkovChainsUnified mcu(draws, chain_sizes);
  Eigen::Ref<const Eigen::VectorXd> col = mcu.draws(0);
  EXPECT_EQ(col.data(), draws.col(0).data());  // test memory sharing
}

TEST(MarkovChainsUnified, DrawsThrowsOnOutOfRangeDimension) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnuts::MarkovChainsUnified mcu(draws, chain_sizes);
  EXPECT_THROW(mcu.draws(-1), std::out_of_range);
  EXPECT_THROW(mcu.draws(2), std::out_of_range);
  EXPECT_THROW(mcu.draws(100), std::out_of_range);
}

// edge case

TEST(MarkovChainsUnified, SingleChainSingleDrawSingleDimension) {
  Eigen::MatrixXd draws(1, 1);
  draws << 42.0;
  walnuts::MarkovChainsUnified mcu(draws, {1});
  EXPECT_EQ(mcu.num_chains(), std::size_t{1});
  EXPECT_EQ(mcu.num_draws(), std::size_t{1});
  EXPECT_EQ(mcu.dims(), std::size_t{1});
  EXPECT_EQ(mcu.min_chain_size(), Eigen::Index{1});
  EXPECT_DOUBLE_EQ(mcu.draws(0)(0), 42.0);
}

// mean() function

TEST(Mean, MarkovChainsSplitReturnsCorrectMean) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd m = walnuts::mean(mcs);
  ASSERT_EQ(m.size(), 2);
  EXPECT_DOUBLE_EQ(m(0), 8.0);
  EXPECT_DOUBLE_EQ(m(1), 9.0);
}

TEST(Mean, MarkovChainsUnifiedReturnsCorrectMean) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnuts::MarkovChainsUnified mcu(draws, chain_sizes);
  Eigen::RowVectorXd m = walnuts::mean(mcu);
  ASSERT_EQ(m.size(), 2);
  EXPECT_DOUBLE_EQ(m(0), 8.0);
  EXPECT_DOUBLE_EQ(m(1), 9.0);
}

TEST(Mean, SingleDrawReturnsDrawValue) {
  std::vector<Eigen::MatrixXd> one;
  Eigen::MatrixXd c(1, 1);
  c << 7.5;
  one.push_back(std::move(c));
  walnuts::MarkovChainsSplit single(one);
  Eigen::RowVectorXd m = walnuts::mean(single);
  ASSERT_EQ(m.size(), 1);
  EXPECT_DOUBLE_EQ(m(0), 7.5);
}

TEST(Mean, ResultSizeMatchesDims) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  EXPECT_EQ(static_cast<std::size_t>(walnuts::mean(mcs).size()), mcs.dims());
}

// sample_variance() function

TEST(SampleVariance, ThrowsOnSingleDraw) {
  std::vector<Eigen::MatrixXd> one;
  Eigen::MatrixXd c(1, 2);
  c << 1.0, 2.0;
  one.push_back(std::move(c));
  walnuts::MarkovChainsSplit mcs(one);
  EXPECT_THROW(walnuts::sample_variance(mcs), std::domain_error);
}

TEST(SampleVariance, MarkovChainsSplitReturnsCorrectVariance) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd v = walnuts::sample_variance(mcs);
  ASSERT_EQ(v.size(), 2);
  EXPECT_DOUBLE_EQ(v(0), 24.0);
  EXPECT_DOUBLE_EQ(v(1), 24.0);
}

TEST(SampleVariance, MarkovChainsUnifiedReturnsCorrectVariance) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnuts::MarkovChainsUnified mcu(draws, chain_sizes);
  Eigen::RowVectorXd v = walnuts::sample_variance(mcu);
  ASSERT_EQ(v.size(), 2);
  EXPECT_DOUBLE_EQ(v(0), 24.0);
  EXPECT_DOUBLE_EQ(v(1), 24.0);
}

TEST(SampleVariance, BothTypesAgreeOnSameData) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::MatrixXd draws = make_unified_draws();
  walnuts::MarkovChainsUnified mcu(draws, chain_sizes);
  EXPECT_TRUE(walnuts::sample_variance(mcs).isApprox(
      walnuts::sample_variance(mcu)));
}

TEST(SampleVariance, ResultSizeMatchesDims) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  EXPECT_EQ(static_cast<std::size_t>(walnuts::sample_variance(mcs).size()),
            mcs.dims());
}

TEST(SampleVariance, TwoIdenticalDrawsGivesZeroVariance) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(2, 2);
  c << 3.0, 7.0,
       3.0, 7.0;
  chains.push_back(std::move(c));
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd v = walnuts::sample_variance(mcs);
  ASSERT_EQ(v.size(), 2);
  EXPECT_DOUBLE_EQ(v(0), 0.0);
  EXPECT_DOUBLE_EQ(v(1), 0.0);
}

TEST(SampleVariance, TwoDrawsMatchesHandCalculation) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(2, 1);
  c << 1.0, 3.0;
  chains.push_back(std::move(c));
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd v = walnuts::sample_variance(mcs);
  ASSERT_EQ(v.size(), 1);
  EXPECT_DOUBLE_EQ(v(0), 2.0);
}

// sample_standard_deviation() function

TEST(SampleStandardDeviation, ThrowsOnSingleDraw) {
  std::vector<Eigen::MatrixXd> one;
  Eigen::MatrixXd c(1, 2);
  c << 1.0, 2.0;
  one.push_back(std::move(c));
  walnuts::MarkovChainsSplit mcs(one);
  EXPECT_THROW(walnuts::sample_standard_deviation(mcs), std::domain_error);
}

TEST(SampleStandardDeviation, MarkovChainsSplitReturnsCorrectStdDev) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd sd = walnuts::sample_standard_deviation(mcs);
  ASSERT_EQ(sd.size(), 2);
  EXPECT_DOUBLE_EQ(sd(0), std::sqrt(24.0));
  EXPECT_DOUBLE_EQ(sd(1), std::sqrt(24.0));
}

TEST(SampleStandardDeviation, MarkovChainsUnifiedReturnsCorrectStdDev) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnuts::MarkovChainsUnified mcu(draws, chain_sizes);
  Eigen::RowVectorXd sd = walnuts::sample_standard_deviation(mcu);
  ASSERT_EQ(sd.size(), 2);
  EXPECT_DOUBLE_EQ(sd(0), std::sqrt(24.0));
  EXPECT_DOUBLE_EQ(sd(1), std::sqrt(24.0));
}

TEST(SampleStandardDeviation, IsSquareRootOfSampleVariance) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd sd = walnuts::sample_standard_deviation(mcs);
  Eigen::RowVectorXd v = walnuts::sample_variance(mcs);
  EXPECT_TRUE(sd.array().square().matrix().isApprox(v));
}

TEST(SampleStandardDeviation, TwoIdenticalDrawsGivesZeroStdDev) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(2, 1);
  c << 5.0, 5.0;
  chains.push_back(std::move(c));
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd sd = walnuts::sample_standard_deviation(mcs);
  ASSERT_EQ(sd.size(), 1);
  EXPECT_DOUBLE_EQ(sd(0), 0.0);
}

// quantiles() function

// Generate reference values with Python.

// import numpy as np
// # data matches examples in this test file
// col0 = np.array([1, 3, 5, 7, 9, 11, 13, 15], dtype=float)
// col1 = np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=float)
// for p in [0.0, 0.1, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0]:
//     q0 = np.quantile(col0, p, method='linear')
//     q1 = np.quantile(col1, p, method='linear')
//     print(f'p={p:.2f} → ({q0:.4f}, {q1:.4f})')
// doc = np.array([9, 11, 5, 3], dtype=float)
// print(f'\nDoc example p=0.6: {np.quantile(doc_eg, 0.6, method="linear")}')
// output:
//   p=0.00 → (1.0,   2.0)
//   p=0.25 → (4.5,   5.5)
//   p=0.50 → (8.0,   9.0)
//   p=0.75 → (11.5, 12.5)
//   p=1.00 → (15.0, 16.0)
//   p=0.10 → (2.4,   3.4)
//   p=0.90 → (13.6, 14.6)
//   p=0.60 → (9.4,  10.4)

// constructor

TEST(Quantiles, ThrowsOnProbBelowZero) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(2);
  probs << -0.1, 0.5;
  EXPECT_THROW(walnuts::quantiles(mcs, probs), std::invalid_argument);
}

TEST(Quantiles, ThrowsOnProbAboveOne) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(2);
  probs << 0.5, 1.1;
  EXPECT_THROW(walnuts::quantiles(mcs, probs), std::invalid_argument);
}

TEST(Quantiles, ThrowsOnNaNProb) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(2);
  probs << 0.5, std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(walnuts::quantiles(mcs, probs), std::invalid_argument);
}

// output shape

TEST(Quantiles, EmptyProbsReturnsEmptyMatrix) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(0);
  Eigen::MatrixXd result = walnuts::quantiles(mcs, probs);
  EXPECT_EQ(result.rows(), 0);
  EXPECT_EQ(result.cols(), static_cast<Eigen::Index>(mcs.dims()));
}

TEST(Quantiles, ResultShapeIsProbs_x_Dims) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(3);
  probs << 0.25, 0.5, 0.75;
  Eigen::MatrixXd result = walnuts::quantiles(mcs, probs);
  EXPECT_EQ(result.rows(), Eigen::Index{3});
  EXPECT_EQ(result.cols(), Eigen::Index{2});
}

// boundary cases of probabilities

TEST(Quantiles, ProbZeroReturnsMinimum) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(1);
  probs << 0.0;
  Eigen::MatrixXd result = walnuts::quantiles(mcs, probs);
  EXPECT_DOUBLE_EQ(result(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(result(0, 1), 2.0);
}

TEST(Quantiles, ProbOneReturnsMaximum) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(1);
  probs << 1.0;
  Eigen::MatrixXd result = walnuts::quantiles(mcs, probs);
  EXPECT_DOUBLE_EQ(result(0, 0), 15.0);
  EXPECT_DOUBLE_EQ(result(0, 1), 16.0);
}

// reference values match numpy

TEST(Quantiles, QuartilesMatchNumpy) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(5);
  probs << 0.0, 0.25, 0.5, 0.75, 1.0;
  Eigen::MatrixXd result = walnuts::quantiles(mcs, probs);
  Eigen::MatrixXd expected(5, 2);
  expected <<  1.0,  2.0,
               4.5,  5.5,
               8.0,  9.0,
              11.5, 12.5,
              15.0, 16.0;
  EXPECT_TRUE(result.isApprox(expected));
}

TEST(Quantiles, InteriorProbsMatchNumpy) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(3);
  probs << 0.1, 0.6, 0.9;
  Eigen::MatrixXd result = walnuts::quantiles(mcs, probs);
  Eigen::MatrixXd expected(3, 2);
  expected <<  2.4,  3.4,
               9.4, 10.4,
              13.6, 14.6;
  EXPECT_TRUE(result.isApprox(expected));
}

// example from the documentation

TEST(Quantiles, DocExampleMatchesPseudocode) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(4, 1);
  c << 9.0, 11.0, 5.0, 3.0;
  chains.push_back(std::move(c));
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(1);
  probs << 0.6;
  Eigen::MatrixXd result = walnuts::quantiles(mcs, probs);
  EXPECT_DOUBLE_EQ(result(0, 0), 8.2);
}

// split and unified both work

TEST(Quantiles, BothTypesAgreeOnSameData) {
  auto chains = make_example_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::MatrixXd draws = make_unified_draws();
  walnuts::MarkovChainsUnified mcu(draws, chain_sizes);
  Eigen::VectorXd probs(5);
  probs << 0.0, 0.25, 0.5, 0.75, 1.0;
  EXPECT_TRUE(walnuts::quantiles(mcs, probs).isApprox(
      walnuts::quantiles(mcu, probs)));
}

// autocovariance() function

// import numpy as np
// def autocovariance_chain(chain):
//     N = chain.shape[0]
//     result = np.zeros_like(chain)
//     for d in range(chain.shape[1]):
//         col = chain[:, d]
//         mu = col.mean()
//         centered = col - mu
//         for lag in range(N):
// 	    # divide by N for MLE here
//             result[lag, d] = np.sum(centered[:N-lag] * centered[lag:]) / N
//     return result
// # Non-degenerate example
// ca = np.array([[1, 2], [4, 6]], dtype=float)
// cb = np.array([[3, 8], [7, 1], [2, 9]], dtype=float)
// cc = np.array([[6, 4], [1, 7], [8, 2]], dtype=float)

std::vector<Eigen::MatrixXd> make_acov_chains() {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd ca(2, 2);
  ca << 1, 2,
        4, 6;
  Eigen::MatrixXd cb(3, 2);
  cb << 3, 8,
        7, 1,
        2, 9;
  Eigen::MatrixXd cc(3, 2);
  cc << 6, 4,
        1, 7,
        8, 2;
  chains.push_back(std::move(ca));
  chains.push_back(std::move(cb));
  chains.push_back(std::move(cc));
  return chains;
}

Eigen::MatrixXd autocovariance_direct(const Eigen::MatrixXd& chain) {
  const Eigen::Index N = chain.rows();
  const Eigen::Index D = chain.cols();
  Eigen::RowVectorXd mu = chain.colwise().mean();
  Eigen::MatrixXd centered = chain.rowwise() - mu;
  Eigen::MatrixXd result(N, D);
  for (Eigen::Index lag = 0; lag < N; ++lag) {
    result.row(lag) =
        (centered.topRows(N - lag).array() *
         centered.bottomRows(N - lag).array())
            .colwise()
            .sum() /
        static_cast<double>(N);
  }
  return result;
}

// shape of output

TEST(Autocovariance, ResultShapeIsNumDrawsTimesDims) {
  auto chains = make_acov_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::MatrixXd acov = walnuts::autocovariance(mcs);
  EXPECT_EQ(acov.rows(), static_cast<Eigen::Index>(mcs.num_draws()));
  EXPECT_EQ(acov.cols(), static_cast<Eigen::Index>(mcs.dims()));
}

// lag zero

TEST(Autocovariance, LagZeroEqualsChainBiasedVariance) {
  auto chains = make_acov_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::MatrixXd acov = walnuts::autocovariance(mcs);
  // chain 0 starts at row 0: biased var of [1,4]=9/4, [2,6]=4
  EXPECT_NEAR(acov(0, 0),  9.0/4.0,  1e-10);
  EXPECT_NEAR(acov(0, 1),  4.0,      1e-10);
  // chain 1 starts at row 2: biased var of [3,7,2]=14/3, [8,1,9]=38/3
  EXPECT_NEAR(acov(2, 0), 14.0/3.0,  1e-10);
  EXPECT_NEAR(acov(2, 1), 38.0/3.0,  1e-10);
  // chain 2 starts at row 5: biased var of [6,1,8]=26/3, [4,7,2]=38/9
  EXPECT_NEAR(acov(5, 0), 26.0/3.0,  1e-10);
  EXPECT_NEAR(acov(5, 1), 38.0/9.0,  1e-10);
 }

// non-degenerate example

TEST(Autocovariance, FullResultMatchesReference) {
  auto chains = make_acov_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::MatrixXd acov = walnuts::autocovariance(mcs);

  Eigen::MatrixXd expected(8, 2);
  expected <<   9.0/4.0,   4.0,        // chain 0, lag 0
               -9.0/8.0,  -2.0,        // chain 0, lag 1
               14.0/3.0,  38.0/3.0,    // chain 1, lag 0
               -3.0,      -25.0/3.0,   // chain 1, lag 1
                2.0/3.0,   2.0,         // chain 1, lag 2
               26.0/3.0,  38.0/9.0,    // chain 2, lag 0
              -16.0/3.0,  -64.0/27.0,  // chain 2, lag 1
                1.0,        7.0/27.0;  // chain 2, lag 2

  EXPECT_TRUE(acov.isApprox(expected, 1e-10));
}

// check FFT vs. direct

TEST(Autocovariance, MatchesDirectComputationPerChain) {
  auto chains = make_acov_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::MatrixXd acov = walnuts::autocovariance(mcs);
  std::vector<Eigen::Index> starts{0, 2, 5};
  std::vector<Eigen::Index> sizes{2, 3, 3};
  for (std::size_t m = 0; m < 3; ++m) {
    Eigen::MatrixXd chain = mcs.chain_view(m);
    Eigen::MatrixXd direct = autocovariance_direct(chain);
    Eigen::MatrixXd fft_block = acov.middleRows(starts[m], sizes[m]);
    EXPECT_TRUE(fft_block.isApprox(direct, 1e-10))
        << "Mismatch at chain " << m;
  }
}

// check both coding types

TEST(Autocovariance, BothTypesAgreeOnSameData) {
  auto chains = make_acov_chains();
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::MatrixXd draws(8, 2);
  draws << 1, 2,
           4, 6,
           3, 8,
           7, 1,
           2, 9,
           6, 4,
           1, 7,
           8, 2;
  walnuts::MarkovChainsUnified mcu(draws, {2, 3, 3});
  EXPECT_TRUE(walnuts::autocovariance(mcs).isApprox(
      walnuts::autocovariance(mcu), 1e-10));
}

// single draw chain

TEST(Autocovariance, SingleDrawChainGivesZero) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(1, 2);
  c << 3.0, 7.0;
  chains.push_back(std::move(c));
  walnuts::MarkovChainsSplit mcs(chains);
  Eigen::MatrixXd acov = walnuts::autocovariance(mcs);
  ASSERT_EQ(acov.rows(), Eigen::Index{1});
  EXPECT_NEAR(acov(0, 0), 0.0, 1e-10);
  EXPECT_NEAR(acov(0, 1), 0.0, 1e-10);
}
