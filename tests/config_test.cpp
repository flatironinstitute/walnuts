#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <walnuts.hpp>

static std::vector<double> inf_nan() {
  std::vector<double> result;
  result.push_back(std::numeric_limits<double>::quiet_NaN());
  result.push_back(std::numeric_limits<double>::infinity());
  return result;
}

static std::vector<double> inf_nan_neg() {
  std::vector<double> result = inf_nan();
  result.push_back(-0.3);
  return result;
}

static std::vector<double> inf_nan_neg_zero() {
  std::vector<double> result = inf_nan_neg();
  result.push_back(0.0);
  return result;
}

static std::vector<double> inf_nan_neg_zero_geq_one() {
  std::vector<double> result = inf_nan_neg_zero();
  result.push_back(1.0);
  result.push_back(1.5);
  return result;
}

static std::vector<double> inf_nan_neg_zero_leq_one() {
  std::vector<double> result = inf_nan_neg_zero();
  result.push_back(1.0);
  result.push_back(0.99);
  return result;
}

static void std_normal(const Eigen::VectorXd& x, double& lp,
                       Eigen::VectorXd& grad) {
  lp = -0.5 * x.dot(x);
  grad = -x;
}

// class InitChainConfig ********************************************

TEST(InitChainConfig, ConstructorStoresStepSize) {
  Eigen::VectorXd position(2);
  position << 1.0, 2.0;
  Eigen::VectorXd mass(2);
  mass << 0.5, 1.5;
  walnuts::InitChainConfig cfg(0.1, position, mass);
  EXPECT_DOUBLE_EQ(cfg.step_size(), 0.1);
}

TEST(InitChainConfig, ConstructorStoresPosition) {
  Eigen::VectorXd position(2);
  position << 1.0, 2.0;
  Eigen::VectorXd mass(2);
  mass << 0.5, 1.5;
  walnuts::InitChainConfig cfg(0.1, position, mass);
  ASSERT_EQ(cfg.position().size(), Eigen::Index{2});
  EXPECT_DOUBLE_EQ(cfg.position()(0), 1.0);
  EXPECT_DOUBLE_EQ(cfg.position()(1), 2.0);
}

TEST(InitChainConfig, ConstructorStoresMass) {
  Eigen::VectorXd position(2);
  position << 1.0, 2.0;
  Eigen::VectorXd mass(2);
  mass << 0.5, 1.5;
  walnuts::InitChainConfig cfg(0.1, position, mass);
  ASSERT_EQ(cfg.mass().size(), Eigen::Index{2});
  EXPECT_DOUBLE_EQ(cfg.mass()(0), 0.5);
  EXPECT_DOUBLE_EQ(cfg.mass()(1), 1.5);
}

TEST(InitChainConfig, PositionIsReturnedByReference) {
  Eigen::VectorXd position(2);
  position << 1.0, 2.0;
  Eigen::VectorXd mass(2);
  mass << 0.5, 1.5;
  walnuts::InitChainConfig cfg(0.1, position, mass);
  EXPECT_EQ(&cfg.position(), &cfg.position());
}

TEST(InitChainConfig, MassIsReturnedByReference) {
  Eigen::VectorXd position(2);
  position << 1.0, 2.0;
  Eigen::VectorXd mass(2);
  mass << 0.5, 1.5;
  walnuts::InitChainConfig cfg(0.1, position, mass);
  EXPECT_EQ(&cfg.mass(), &cfg.mass());
}


// classes InitConfig and InitConfigBuilder *************************

// default InitConfig 

TEST(InitConfigBuilder, DefaultsAreCorrect) {
  Eigen::Index D_Eigen = Eigen::Index{2};
  std::size_t D = 2;
  std::size_t M = 3;
  walnuts::InitConfig cfg = walnuts::InitConfigBuilder(M, D).build();
  EXPECT_EQ(cfg.num_chains(), M);
  EXPECT_EQ(cfg.dims(), D);
  for (std::size_t m = 0; m < M; ++m) {
    EXPECT_DOUBLE_EQ(cfg.step_size(m), 0.1);  // default step size 0.1
    EXPECT_TRUE(cfg.position(m).isZero()); // default position 0
    EXPECT_EQ(cfg.position(m).size(), D_Eigen);
    EXPECT_TRUE(cfg.mass(m).isOnes());  // default mass I
    EXPECT_EQ(cfg.mass(m).size(), D_Eigen);
  }
}

TEST(InitConfigBuilder, EmptyConfigHasZeroDims) {
  walnuts::InitConfig cfg = walnuts::InitConfigBuilder(0, 0).build();
  EXPECT_EQ(cfg.num_chains(), std::size_t{0});
  EXPECT_EQ(cfg.dims(), std::size_t{0});
  EXPECT_TRUE(cfg.step_sizes().empty());
  EXPECT_TRUE(cfg.positions().empty());
  EXPECT_TRUE(cfg.masses().empty());
}

// step size (double)

TEST(InitConfigBuilder, ScalarStepSizeSetsAllChains) {
  walnuts::InitConfig cfg =
      walnuts::InitConfigBuilder(3, 2).step_sizes(0.5).build();
  EXPECT_EQ(cfg.step_sizes().size(), std::size_t{3});
  for (std::size_t n = 0; n < 3; ++n) {
    EXPECT_DOUBLE_EQ(cfg.step_size(n), 0.5);
  }
}

TEST(InitConfigBuilder, ScalarStepSizeThrowsOnNonPositive) {
  for (auto x : inf_nan_neg_zero()) {
    walnuts::InitConfigBuilder b(3, 2);
    EXPECT_THROW(b.step_sizes(x), std::invalid_argument);
  }
}

// step size (vector)

TEST(InitConfigBuilder, VectorStepSizesSetPerChain) {
  std::vector<double> sizes{0.1, 0.2, 0.3};
  walnuts::InitConfig cfg =
      walnuts::InitConfigBuilder(3, 2).step_sizes(sizes).build();
  for (std::size_t n = 0; n < 3; ++n) {
    EXPECT_DOUBLE_EQ(cfg.step_size(n), sizes[n]);
  }
}

TEST(InitConfigBuilder, VectorStepSizesThrowsOnWrongSize) {
  walnuts::InitConfigBuilder b(3, 2);
  std::vector<double> wrong_num_chains{0.1, 0.2};
  EXPECT_THROW(b.step_sizes(wrong_num_chains), std::invalid_argument);
}

TEST(InitConfigBuilder, VectorStepSizesThrowsOnNonPositiveElement) {
  for (auto x : inf_nan_neg_zero()) {
    walnuts::InitConfigBuilder b(3, 2);
    std::vector<double> bad{0.1, x, 0.3};
    EXPECT_THROW(b.step_sizes(bad), std::invalid_argument);

    walnuts::InitConfigBuilder b2(3, 2);
    std::vector<double> bad2{x, 0.1, 0.3};
    EXPECT_THROW(b2.step_sizes(bad2), std::invalid_argument);
  }
}

// positions (VectorXd)

TEST(InitConfigBuilder, ScalarPositionSetsAllChains) {
  Eigen::VectorXd pos(2);
  pos << 3.0, 4.0;
  walnuts::InitConfig cfg =
      walnuts::InitConfigBuilder(3, 2).positions(pos).build();
  for (std::size_t n = 0; n < 3; ++n) {
    EXPECT_TRUE(cfg.position(n).isApprox(pos));
  }
}

TEST(InitConfigBuilder, ScalarPositionThrowsOnWrongDims) {
  walnuts::InitConfigBuilder b(3, 2);
  Eigen::VectorXd wrong_dims(3);
  wrong_dims << 1.0, 2.0, 3.0;
  EXPECT_THROW(b.positions(wrong_dims), std::invalid_argument);
}

TEST(InitConfigBuilder, ScalarPositionThrowsOnNonFinite) {
  for (auto x : inf_nan()) {
    walnuts::InitConfigBuilder b(3, 2);
    Eigen::VectorXd non_finite(2);
    non_finite << 1.0, x;
    EXPECT_THROW(b.positions(non_finite), std::invalid_argument);
  }
}

// positions (vector<VectorXd>)

TEST(InitConfigBuilder, VectorPositionsSetsPerChain) {
  std::vector<Eigen::VectorXd> vs(3, Eigen::VectorXd(2));
  vs[0] << 1.0, 2.0;
  vs[1] << 3.0, 4.0;
  vs[2] << 5.0, 6.0;
  walnuts::InitConfig cfg =
      walnuts::InitConfigBuilder(3, 2).positions(vs).build();
  for (std::size_t m = 0; m < 3; ++m) {
    EXPECT_TRUE(cfg.position(m).isApprox(vs[m]));
  }
}

TEST(InitConfigBuilder, VectorPositionsThrowsOnWrongNumberOfChains) {
  walnuts::InitConfigBuilder b(3, 2);
  std::vector<Eigen::VectorXd> wrong_chains(2, Eigen::VectorXd::Zero(2));
  EXPECT_THROW(b.positions(wrong_chains), std::invalid_argument);
}

TEST(InitConfigBuilder, VectorPositionsThrowsOnWrongDims) {
  walnuts::InitConfigBuilder b(3, 2);
  std::vector<Eigen::VectorXd> wrong_dims(3, Eigen::VectorXd::Zero(3));
  EXPECT_THROW(b.positions(wrong_dims), std::invalid_argument);
}

TEST(InitConfigBuilder, VectorPositionsThrowsOnNonFinite) {
  for (auto x : inf_nan()) {
    walnuts::InitConfigBuilder b(3, 2);
    std::vector<Eigen::VectorXd> non_finite(3, Eigen::VectorXd::Zero(2));
    non_finite[1](0) = x;
    EXPECT_THROW(b.positions(non_finite), std::invalid_argument);
  }
}

// positions (vector<VectorXd>&&)

TEST(InitConfigBuilder, MovePositionsSetsPerChain) {
  std::vector<Eigen::VectorXd> vs(2, Eigen::VectorXd(2));
  vs[0] << 1.0, 2.0;
  vs[1] << 3.0, 4.0;
  std::vector<Eigen::VectorXd> vs_expected = vs;
  walnuts::InitConfig cfg =
      walnuts::InitConfigBuilder(2, 2).positions(std::move(vs)).build();
  for (std::size_t m = 0; m < 2; ++m) {
    EXPECT_TRUE(cfg.position(m).isApprox(vs_expected[m]));
  }
}

TEST(InitConfigBuilder, MovePositionsThrowsOnWrongNumberOfChains) {
  walnuts::InitConfigBuilder b(3, 2);
  std::vector<Eigen::VectorXd> wrong_chains(2, Eigen::VectorXd::Zero(2));
  EXPECT_THROW(b.positions(std::move(wrong_chains)), std::invalid_argument);
}

TEST(InitConfigBuilder, MovePositionsThrowsOnWrongDims) {
  walnuts::InitConfigBuilder b(3, 2);
  std::vector<Eigen::VectorXd> wrong_dims(3, Eigen::VectorXd::Zero(3));
  EXPECT_THROW(b.positions(std::move(wrong_dims)), std::invalid_argument);
}

TEST(InitConfigBuilder, MovePositionsThrowsOnNonFinite) {
  for (auto x : inf_nan()) {
    walnuts::InitConfigBuilder b(3, 2);
    std::vector<Eigen::VectorXd> bad(3, Eigen::VectorXd::Zero(2));
    bad[0](1) = x;
    EXPECT_THROW(b.positions(std::move(bad)), std::invalid_argument);
  }
}

// positions (RNG, scale)

TEST(InitConfigBuilder, RandomPositionsHaveCorrectShape) {
  std::mt19937 rng(139872);
  walnuts::InitConfig cfg =
      walnuts::InitConfigBuilder(3, 4).positions(rng, 1.0).build();
  EXPECT_EQ(cfg.positions().size(), std::size_t{3});
  for (std::size_t n = 0; n < 3; ++n) {
    EXPECT_EQ(cfg.position(n).size(), Eigen::Index{4});
  }
}

TEST(InitConfigBuilder, RandomPositionsScaledByInitScale) {
  std::mt19937 rng1(876), rng2(876);
  walnuts::InitConfig cfg1 =
      walnuts::InitConfigBuilder(2, 3).positions(rng1, 1.0).build();
  walnuts::InitConfig cfg2 =
      walnuts::InitConfigBuilder(2, 3).positions(rng2, 2.0).build();
  for (std::size_t n = 0; n < 2; ++n) {
    EXPECT_TRUE(cfg2.position(n).isApprox(2.0 * cfg1.position(n)));
  }
}

TEST(InitConfigBuilder, RandomPositionsThrowsOnNonPositiveScale) {
  for (auto x : inf_nan_neg_zero()) {
    std::mt19937 rng(58375232);
    walnuts::InitConfigBuilder b(3, 2);
    EXPECT_THROW(b.positions(rng, x), std::invalid_argument);
  }
}

// masses (VectorXd)

TEST(InitConfigBuilder, ScalarMassSetsAllChains) {
  Eigen::VectorXd mass(2);
  mass << 2.0, 3.0;
  walnuts::InitConfig cfg =
      walnuts::InitConfigBuilder(3, 2).masses(mass).build();
  for (std::size_t n = 0; n < 3; ++n) {
    EXPECT_TRUE(cfg.mass(n).isApprox(mass));
  }
}

TEST(InitConfigBuilder, ScalarMassThrowsOnWrongDims) {
  walnuts::InitConfigBuilder b(3, 2);
  Eigen::VectorXd wrong_dims(3);
  wrong_dims << 1.0, 2.0, 3.0;
  EXPECT_THROW(b.masses(wrong_dims), std::invalid_argument);
}

TEST(InitConfigBuilder, ScalarMassThrowsOnNonPositiveElement) {
  for (auto x : inf_nan_neg_zero()) {
    walnuts::InitConfigBuilder b(3, 2);
    Eigen::VectorXd bad(2);
    bad << 1.0, x;
    EXPECT_THROW(b.masses(bad), std::invalid_argument);
    bad << x, 2.9;
    EXPECT_THROW(b.masses(bad), std::invalid_argument);
  }
}

// masses (vector<VectorXd>) const ref

TEST(InitConfigBuilder, VectorMassesSetsPerChain) {
  std::vector<Eigen::VectorXd> vs(3, Eigen::VectorXd(2));
  vs[0] << 1.0, 2.0;
  vs[1] << 3.0, 4.0;
  vs[2] << 5.0, 6.0;
  walnuts::InitConfig cfg =
      walnuts::InitConfigBuilder(3, 2).masses(vs).build();
  for (std::size_t m = 0; m < 3; ++m) {
    EXPECT_TRUE(cfg.mass(m).isApprox(vs[m]));
  }
}


TEST(InitConfigBuilder, VectorMassesThrowsOnWrongNumberOfChains) {
  walnuts::InitConfigBuilder b(3, 2);
  std::vector<Eigen::VectorXd> wrong_chains(2, Eigen::VectorXd::Ones(2));
  EXPECT_THROW(b.masses(wrong_chains), std::invalid_argument);
}

TEST(InitConfigBuilder, VectorMassesThrowsOnWrongDims) {
  walnuts::InitConfigBuilder b(3, 2);
  std::vector<Eigen::VectorXd> wrong_dims(3, Eigen::VectorXd::Ones(3));
  EXPECT_THROW(b.masses(wrong_dims), std::invalid_argument);
}

TEST(InitConfigBuilder, VectorMassesThrowsOnNonPositive) {
  for (auto x : inf_nan_neg_zero()) {
    walnuts::InitConfigBuilder b(3, 2);
    std::vector<Eigen::VectorXd> bad(3, Eigen::VectorXd::Ones(2));
    bad[1](0) = x;
    EXPECT_THROW(b.masses(bad), std::invalid_argument);
  }
}

// masses (vector<VectorXd>&&)

TEST(InitConfigBuilder, MoveMassesSetsPerChain) {
  std::vector<Eigen::VectorXd> vs(2, Eigen::VectorXd(2));
  vs[0] << 1.0, 2.0;
  vs[1] << 3.0, 4.0;
  std::vector<Eigen::VectorXd> expected = vs;

  walnuts::InitConfig cfg =
      walnuts::InitConfigBuilder(2, 2).masses(std::move(vs)).build();
  for (std::size_t n = 0; n < 2; ++n) {
    EXPECT_TRUE(cfg.mass(n).isApprox(expected[n]));
  }
}

TEST(InitConfigBuilder, MoveMassesThrowsOnWrongNumberOfChains) {
  walnuts::InitConfigBuilder b(3, 2);
  std::vector<Eigen::VectorXd> wrong_chains(2, Eigen::VectorXd::Ones(2));
  EXPECT_THROW(b.masses(std::move(wrong_chains)), std::invalid_argument);
}

TEST(InitConfigBuilder, MoveMassesThrowsOnWrongDims) {
  walnuts::InitConfigBuilder b(3, 2);
  std::vector<Eigen::VectorXd> wrong_dims(3, Eigen::VectorXd::Ones(3));
  EXPECT_THROW(b.masses(std::move(wrong_dims)), std::invalid_argument);
}

TEST(InitConfigBuilder, MoveMassesThrowsOnNonPositiveElement) {
  for (auto x : inf_nan_neg_zero()) {
    walnuts::InitConfigBuilder b(3, 2);
    std::vector<Eigen::VectorXd> bad(3, Eigen::VectorXd::Ones(2));
    bad[2](1) = x;
    EXPECT_THROW(b.masses(std::move(bad)), std::invalid_argument);
  }
}

// masses(logp_grad, smoothing)

TEST(InitConfigBuilder, LogpGradMassesHaveCorrectShape) {
  Eigen::VectorXd pos(2);
  pos << 1.0, 2.0;
  walnuts::InitConfig cfg = walnuts::InitConfigBuilder(3, 2)
                                .positions(pos)
                                .masses(std_normal, 0.5)
                                .build();
  EXPECT_EQ(cfg.masses().size(), std::size_t{3});
  for (std::size_t n = 0; n < 3; ++n) {
    EXPECT_EQ(cfg.mass(n).size(), Eigen::Index{2});
  }
}

TEST(InitConfigBuilder, LogpGradMassesMatchHandCalculation) {
  // pos = [1, 2];  grad = [-1, -2];  smooth = 0.5
  // mass = (1 - smooth) * sqrt(abs(grad)) + smooth
  //      = 0.5 * [sqrt(1), sqrt(2)] + 0.5 = [1, 0.5 * sqrt(2) + 0.5]
  Eigen::VectorXd pos(2);
  pos << 1.0, 2.0;
  const double s = 0.5;
  walnuts::InitConfig cfg = walnuts::InitConfigBuilder(1, 2)
                                .positions(pos)
                                .masses(std_normal, s)
                                .build();
  Eigen::VectorXd expected(2);
  expected(0) = (1 - s) * std::sqrt(1.0) + s;
  expected(1) = (1 - s) * std::sqrt(2.0) + s;
  EXPECT_TRUE(cfg.mass(0).isApprox(expected));
}

TEST(InitConfigBuilder, LogpGradMassesOneThrows) {
  Eigen::VectorXd pos(2);
  pos << 100.0, -50.0;
  auto builder = walnuts::InitConfigBuilder(2, 2);
  auto& builder_chain = builder.positions(pos);
  EXPECT_THROW(builder_chain.masses(std_normal, 1.0), std::invalid_argument);
}

TEST(InitConfigBuilder, LogpGradMassesThrowsOnInvalidSmoothing) {
  walnuts::InitConfigBuilder b(2, 2);
  for (auto x : inf_nan_neg()) {
    EXPECT_THROW(b.masses(std_normal, x), std::invalid_argument);
  }
}

// init_chain_config()

TEST(InitConfig, InitChainConfigReturnsCorrectValues) {
  Eigen::VectorXd pos(2);
  pos << 1.0, 2.0;
  Eigen::VectorXd mass(2);
  mass << 3.0, 4.0;
  walnuts::InitConfig cfg = walnuts::InitConfigBuilder(3, 2)
                                .step_sizes(0.25)
                                .positions(pos)
                                .masses(mass)
                                .build();
  walnuts::InitChainConfig cc = cfg.init_chain_config(0);
  EXPECT_DOUBLE_EQ(cc.step_size(), 0.25);
  EXPECT_TRUE(cc.position().isApprox(pos));
  EXPECT_TRUE(cc.mass().isApprox(mass));
}

// chaining 

TEST(InitConfigBuilder, MethodChainingReturnsBuilder) {
  Eigen::VectorXd pos(2);
  pos << 0.5, -0.5;
  Eigen::VectorXd mass(2);
  mass << 3.0, 4.0;
  auto builder = walnuts::InitConfigBuilder(3, 2);

  auto& chained_builder = builder.step_sizes(0.3);
  EXPECT_EQ(&builder, &chained_builder);

  chained_builder = chained_builder.masses(mass);
  EXPECT_EQ(&builder, &chained_builder);

  chained_builder = chained_builder.positions(pos);
  EXPECT_EQ(&builder, &chained_builder);
}

// classes WarmupConfig and WarmupConfigBuilder *********************

// default build

TEST(WarmupConfig, DefaultValuesAreCorrect) {
  walnuts::WarmupConfig cfg = walnuts::WarmupConfigBuilder().build();
  EXPECT_EQ(cfg.min_iter(), std::size_t{50});
  EXPECT_EQ(cfg.max_iter(), std::size_t{1000});
  EXPECT_DOUBLE_EQ(cfg.step_size_converge_tol(), 0.1);
  EXPECT_DOUBLE_EQ(cfg.mass_converge_tol(), 1.0);
  EXPECT_DOUBLE_EQ(cfg.mass_init_count(), 4.0);
  EXPECT_DOUBLE_EQ(cfg.mass_additive_smoothing(), 1e-5);
  EXPECT_DOUBLE_EQ(cfg.max_macro_steps_target(), 15.0);
  EXPECT_DOUBLE_EQ(cfg.step_accept_rate_target(), 0.8);
  EXPECT_DOUBLE_EQ(cfg.step_learning_rate(), 0.05);
  EXPECT_DOUBLE_EQ(cfg.step_gradient_decay(), 0.8);
  EXPECT_DOUBLE_EQ(cfg.step_sq_gradient_decay(), 0.9);
  EXPECT_DOUBLE_EQ(cfg.step_stabilization(), 1e-4);
  EXPECT_DOUBLE_EQ(cfg.step_learn_rate_decay(), 0.5);
  EXPECT_EQ(cfg.publish_stride(), std::size_t{5});
  EXPECT_EQ(cfg.yield_period(), std::size_t{32});
}

// min_max_iter()

TEST(WarmupConfigBuilder, MinMaxIterSetsCorrectly) {
  walnuts::WarmupConfig cfg =
      walnuts::WarmupConfigBuilder().min_max_iter(10, 500).build();
  EXPECT_EQ(cfg.min_iter(), std::size_t{10});
  EXPECT_EQ(cfg.max_iter(), std::size_t{500});
}

TEST(WarmupConfigBuilder, MinMaxIterAllowsEqualMinAndMax) {
  walnuts::WarmupConfig cfg =
      walnuts::WarmupConfigBuilder().min_max_iter(100, 100).build();
  EXPECT_EQ(cfg.min_iter(), std::size_t{100});
  EXPECT_EQ(cfg.max_iter(), std::size_t{100});
}

TEST(WarmupConfigBuilder, MinMaxIterThrowsWhenMinExceedsMax) {
  walnuts::WarmupConfigBuilder b;
  EXPECT_THROW(b.min_max_iter(500, 10), std::invalid_argument);
}

// step_size_converge_tol()

TEST(WarmupConfigBuilder, StepSizeConvergeTolSetsCorrectly) {
  walnuts::WarmupConfig cfg =
      walnuts::WarmupConfigBuilder().step_size_converge_tol(0.05).build();
  EXPECT_DOUBLE_EQ(cfg.step_size_converge_tol(), 0.05);
}

TEST(WarmupConfigBuilder, StepSizeConvergeTolThrowsOnBadValues) {
  for (auto x : inf_nan_neg_zero()) {
    walnuts::WarmupConfigBuilder b;
    EXPECT_THROW(b.step_size_converge_tol(x), std::invalid_argument);
  }
}

// mass_converge_tol()

TEST(WarmupConfigBuilder, MassConvergeTolSetsCorrectly) {
  walnuts::WarmupConfig cfg =
      walnuts::WarmupConfigBuilder().mass_converge_tol(0.5).build();
  EXPECT_DOUBLE_EQ(cfg.mass_converge_tol(), 0.5);
}

TEST(WarmupConfigBuilder, MassConvergeTolThrowsOnBadValues) {
  for (auto x : inf_nan_neg_zero()) {
    walnuts::WarmupConfigBuilder b;
    EXPECT_THROW(b.mass_converge_tol(x), std::invalid_argument);
  }
}

// mass_init_count()

TEST(WarmupConfigBuilder, MassInitCountSetsCorrectly) {
  walnuts::WarmupConfig cfg =
      walnuts::WarmupConfigBuilder().mass_init_count(10.0).build();
  EXPECT_DOUBLE_EQ(cfg.mass_init_count(), 10.0);
}

TEST(WarmupConfigBuilder, MassInitCountThrowsOnBadValues) {
  for (auto x : inf_nan_neg_zero()) {
    walnuts::WarmupConfigBuilder b;
    EXPECT_THROW(b.mass_init_count(x), std::invalid_argument);
  }
}

// mass_additive_smoothing()

TEST(WarmupConfigBuilder, MassAdditiveSmoothingSetsCorrectly) {
  walnuts::WarmupConfig cfg =
      walnuts::WarmupConfigBuilder().mass_additive_smoothing(0.01).build();
  EXPECT_DOUBLE_EQ(cfg.mass_additive_smoothing(), 0.01);
}

TEST(WarmupConfigBuilder, MassAdditiveSmoothingThrowsOnBadValues) {
  for (auto x : inf_nan_neg_zero()) {
    walnuts::WarmupConfigBuilder b;
    EXPECT_THROW(b.mass_additive_smoothing(x), std::invalid_argument);
  }
}

// max_macro_steps_target()

TEST(WarmupConfigBuilder, MaxMacroStepsTargetSetsCorrectly) {
  walnuts::WarmupConfig cfg =
      walnuts::WarmupConfigBuilder().max_macro_steps_target(20.0).build();
  EXPECT_DOUBLE_EQ(cfg.max_macro_steps_target(), 20.0);
}

TEST(WarmupConfigBuilder, MaxMacroStepsTargetThrowsOnBadValues) {
  for (auto x : inf_nan_neg_zero()) {
    walnuts::WarmupConfigBuilder b;
    EXPECT_THROW(b.max_macro_steps_target(x), std::invalid_argument);
  }
}

// step_accept_rate_target()

TEST(WarmupConfigBuilder, StepAcceptRateTargetSetsCorrectly) {
  walnuts::WarmupConfig cfg =
      walnuts::WarmupConfigBuilder().step_accept_rate_target(0.65).build();
  EXPECT_DOUBLE_EQ(cfg.step_accept_rate_target(), 0.65);
}

TEST(WarmupConfigBuilder, StepAcceptRateTargetThrowsOnBadValues) {
  for (auto x : inf_nan_neg_zero_geq_one()) {
    walnuts::WarmupConfigBuilder b;
    EXPECT_THROW(b.step_accept_rate_target(x), std::invalid_argument);
  }
}

// step_learning_rate()

TEST(WarmupConfigBuilder, StepLearningRateSetsCorrectly) {
  walnuts::WarmupConfig cfg =
      walnuts::WarmupConfigBuilder().step_learning_rate(0.1).build();
  EXPECT_DOUBLE_EQ(cfg.step_learning_rate(), 0.1);
}

TEST(WarmupConfigBuilder, StepLearningRateThrowsOnBadValues) {
  for (auto x : inf_nan_neg_zero()) {
    walnuts::WarmupConfigBuilder b;
    EXPECT_THROW(b.step_learning_rate(x), std::invalid_argument);
  }
}

// step_gradient_decay

TEST(WarmupConfigBuilder, StepGradientDecaySetsCorrectly) {
  walnuts::WarmupConfig cfg =
      walnuts::WarmupConfigBuilder().step_gradient_decay(0.9).build();
  EXPECT_DOUBLE_EQ(cfg.step_gradient_decay(), 0.9);
}

TEST(WarmupConfigBuilder, StepGradientDecayThrowsOnBadValues) {
  for (auto x : inf_nan_neg_zero_geq_one()) {
    walnuts::WarmupConfigBuilder b;
    EXPECT_THROW(b.step_gradient_decay(x), std::invalid_argument);
  }
}

// step_sq_gradient_decay

TEST(WarmupConfigBuilder, StepSqGradientDecaySetsCorrectly) {
  walnuts::WarmupConfig cfg =
      walnuts::WarmupConfigBuilder().step_sq_gradient_decay(0.95).build();
  EXPECT_DOUBLE_EQ(cfg.step_sq_gradient_decay(), 0.95);
}

TEST(WarmupConfigBuilder, StepSqGradientDecayThrowsOnBadValues) {
  for (auto x : inf_nan_neg_zero_geq_one()) {
    walnuts::WarmupConfigBuilder b;
    EXPECT_THROW(b.step_sq_gradient_decay(x), std::invalid_argument);
  }
}

// step_stabilization()

TEST(WarmupConfigBuilder, StepStabilizationSetsCorrectly) {
  walnuts::WarmupConfig cfg =
      walnuts::WarmupConfigBuilder().step_stabilization(1e-3).build();
  EXPECT_DOUBLE_EQ(cfg.step_stabilization(), 1e-3);
}

TEST(WarmupConfigBuilder, StepStabilizationThrowsOnBadValues) {
  for (auto x : inf_nan_neg_zero()) {
    walnuts::WarmupConfigBuilder b;
    EXPECT_THROW(b.step_stabilization(x), std::invalid_argument);
  }
}

// step_learn_rate_decay()

TEST(WarmupConfigBuilder, StepLearnRateDecaySetsCorrectly) {
  walnuts::WarmupConfig cfg =
      walnuts::WarmupConfigBuilder().step_learn_rate_decay(0.75).build();
  EXPECT_DOUBLE_EQ(cfg.step_learn_rate_decay(), 0.75);
}

TEST(WarmupConfigBuilder, StepLearnRateDecayThrowsOnBadValues) {
  for (auto x : inf_nan_neg_zero_geq_one()) {
    walnuts::WarmupConfigBuilder b;
    EXPECT_THROW(b.step_learn_rate_decay(x), std::invalid_argument);
  }
}

// publish_stride()

TEST(WarmupConfigBuilder, PublishStrideSetsCorrectly) {
  walnuts::WarmupConfig cfg =
      walnuts::WarmupConfigBuilder().publish_stride(10).build();
  EXPECT_EQ(cfg.publish_stride(), std::size_t{10});
}

TEST(WarmupConfigBuilder, PublishStrideThrowsOnZero) {
  walnuts::WarmupConfigBuilder b;
  EXPECT_THROW(b.publish_stride(0), std::invalid_argument);
}

// yield_period()

TEST(WarmupConfigBuilder, YieldPeriodSetsCorrectly) {
  walnuts::WarmupConfig cfg =
      walnuts::WarmupConfigBuilder().yield_period(64).build();
  EXPECT_EQ(cfg.yield_period(), std::size_t{64});
}

TEST(WarmupConfigBuilder, YieldPeriodThrowsOnZero) {
  walnuts::WarmupConfigBuilder b;
  EXPECT_THROW(b.yield_period(0), std::invalid_argument);
}

// chaining

TEST(WarmupConfigBuilder, ChainReferenceIdentity) {
  walnuts::WarmupConfigBuilder builder;
  auto& builder_chain = builder
    .min_max_iter(25, 200)
    .step_size_converge_tol(0.05)
    .mass_converge_tol(0.5)
    .mass_init_count(2.0)
    .mass_additive_smoothing(1e-4)
    .max_macro_steps_target(10.0)
    .step_accept_rate_target(0.75)
    .step_learning_rate(0.1)
    .step_gradient_decay(0.85)
    .step_sq_gradient_decay(0.95)
    .step_stabilization(1e-3)
    .step_learn_rate_decay(0.6)
    .publish_stride(2)
    .yield_period(16);
  EXPECT_EQ(&builder, &builder_chain);
}

TEST(WarmupConfigBuilder, FullChainProducesCorrectConfig) {
  walnuts::WarmupConfigBuilder builder;
  auto cfg = builder.min_max_iter(25, 200)
      .step_size_converge_tol(0.05)
      .mass_converge_tol(0.5)
      .mass_init_count(2.0)
      .mass_additive_smoothing(1e-4)
      .max_macro_steps_target(10.0)
      .step_accept_rate_target(0.75)
      .step_learning_rate(0.1)
      .step_gradient_decay(0.85)
      .step_sq_gradient_decay(0.95)
      .step_stabilization(1e-3)
      .step_learn_rate_decay(0.6)
      .publish_stride(2)
    .yield_period(16)
    .build();
  EXPECT_EQ(cfg.min_iter(), std::size_t{25});
  EXPECT_EQ(cfg.max_iter(), std::size_t{200});
  EXPECT_DOUBLE_EQ(cfg.step_size_converge_tol(), 0.05);
  EXPECT_DOUBLE_EQ(cfg.mass_converge_tol(), 0.5);
  EXPECT_DOUBLE_EQ(cfg.mass_init_count(), 2.0);
  EXPECT_DOUBLE_EQ(cfg.mass_additive_smoothing(), 1e-4);
  EXPECT_DOUBLE_EQ(cfg.max_macro_steps_target(), 10.0);
  EXPECT_DOUBLE_EQ(cfg.step_accept_rate_target(), 0.75);
  EXPECT_DOUBLE_EQ(cfg.step_learning_rate(), 0.1);
  EXPECT_DOUBLE_EQ(cfg.step_gradient_decay(), 0.85);
  EXPECT_DOUBLE_EQ(cfg.step_sq_gradient_decay(), 0.95);
  EXPECT_DOUBLE_EQ(cfg.step_stabilization(), 1e-3);
  EXPECT_DOUBLE_EQ(cfg.step_learn_rate_decay(), 0.6);
  EXPECT_EQ(cfg.publish_stride(), std::size_t{2});
  EXPECT_EQ(cfg.yield_period(), std::size_t{16});
}

// classes SamplingConfig and SamplingConfigBuilder *****************

// default build

TEST(SamplingConfig, DefaultValuesAreCorrect) {
  walnuts::SamplingConfig cfg = walnuts::SamplingConfigBuilder().build();
  EXPECT_EQ(cfg.min_iter(), std::size_t{50});
  EXPECT_EQ(cfg.max_iter(), std::size_t{1000});
  EXPECT_EQ(cfg.max_trajectory_doublings(), std::size_t{5});
  EXPECT_EQ(cfg.max_step_halvings(), std::size_t{5});
  EXPECT_DOUBLE_EQ(cfg.max_hamiltonian_error(), 0.5);
  EXPECT_EQ(cfg.min_micro_steps(), std::size_t{1});
  EXPECT_DOUBLE_EQ(cfg.rhat_converge_tol(), 1.01);
}

// min_max_iter()

TEST(SamplingConfigBuilder, MinMaxIterSetsCorrectly) {
  walnuts::SamplingConfig cfg =
      walnuts::SamplingConfigBuilder().min_max_iter(10, 500).build();
  EXPECT_EQ(cfg.min_iter(), std::size_t{10});
  EXPECT_EQ(cfg.max_iter(), std::size_t{500});
}

TEST(SamplingConfigBuilder, MinMaxIterAllowsEqualMinAndMax) {
  walnuts::SamplingConfig cfg =
      walnuts::SamplingConfigBuilder().min_max_iter(100, 100).build();
  EXPECT_EQ(cfg.min_iter(), std::size_t{100});
  EXPECT_EQ(cfg.max_iter(), std::size_t{100});
}

TEST(SamplingConfigBuilder, MinMaxIterThrowsWhenMinExceedsMax) {
  walnuts::SamplingConfigBuilder b;
  EXPECT_THROW(b.min_max_iter(500, 10), std::invalid_argument);
}

// max_trajectory_doublings()

TEST(SamplingConfigBuilder, MaxTrajectoryDoublingsSetsCorrectly) {
  walnuts::SamplingConfig cfg =
      walnuts::SamplingConfigBuilder().max_trajectory_doublings(10).build();
  EXPECT_EQ(cfg.max_trajectory_doublings(), std::size_t{10});
}

TEST(SamplingConfigBuilder, MaxTrajectoryDoublingsAllowsZero) {
  walnuts::SamplingConfig cfg =
      walnuts::SamplingConfigBuilder().max_trajectory_doublings(0).build();
  EXPECT_EQ(cfg.max_trajectory_doublings(), std::size_t{0});
}

// max_step_halvings()

TEST(SamplingConfigBuilder, MaxStepHalvingsSetsCorrectly) {
  walnuts::SamplingConfig cfg =
      walnuts::SamplingConfigBuilder().max_step_halvings(8).build();
  EXPECT_EQ(cfg.max_step_halvings(), std::size_t{8});
}

TEST(SamplingConfigBuilder, MaxStepHalvingsAllowsZero) {
  walnuts::SamplingConfig cfg =
      walnuts::SamplingConfigBuilder().max_step_halvings(0).build();
  EXPECT_EQ(cfg.max_step_halvings(), std::size_t{0});
}

// max_hamiltonian_error()

TEST(SamplingConfigBuilder, MaxHamiltonianErrorSetsCorrectly) {
  walnuts::SamplingConfig cfg =
      walnuts::SamplingConfigBuilder().max_hamiltonian_error(1.0).build();
  EXPECT_DOUBLE_EQ(cfg.max_hamiltonian_error(), 1.0);
}

TEST(SamplingConfigBuilder, MaxHamiltonianErrorThrowsOnBadValues) {
  for (auto x : inf_nan_neg_zero()) {
    walnuts::SamplingConfigBuilder b;
    EXPECT_THROW(b.max_hamiltonian_error(x), std::invalid_argument);
  }
}

// min_micro_steps()

TEST(SamplingConfigBuilder, MinMicroStepsSetsCorrectly) {
  walnuts::SamplingConfig cfg =
      walnuts::SamplingConfigBuilder().min_micro_steps(4).build();
  EXPECT_EQ(cfg.min_micro_steps(), std::size_t{4});
}

TEST(SamplingConfigBuilder, MinMicroStepsThrowsOnZero) {
  walnuts::SamplingConfigBuilder b;
  EXPECT_THROW(b.min_micro_steps(0), std::invalid_argument);
}

// rhat_converge_tol()

TEST(SamplingConfigBuilder, RhatConvergeTolSetsCorrectly) {
  walnuts::SamplingConfig cfg =
      walnuts::SamplingConfigBuilder().rhat_converge_tol(1.05).build();
  EXPECT_DOUBLE_EQ(cfg.rhat_converge_tol(), 1.05);
}

TEST(SamplingConfigBuilder, RhatConvergeTolThrowsOnBadValues) {
  for (auto x : inf_nan_neg_zero_leq_one()) {
    walnuts::SamplingConfigBuilder b;
    EXPECT_THROW(b.rhat_converge_tol(x), std::invalid_argument);
  }
}

// chaining

TEST(SamplingConfigBuilder, FullChainProducesCorrectConfig) {
  walnuts::SamplingConfig cfg = walnuts::SamplingConfigBuilder()
      .min_max_iter(25, 200)
      .max_trajectory_doublings(8)
      .max_step_halvings(3)
      .max_hamiltonian_error(1.0)
      .min_micro_steps(2)
      .rhat_converge_tol(1.05)
      .build();
  EXPECT_EQ(cfg.min_iter(),                   std::size_t{25});
  EXPECT_EQ(cfg.max_iter(),                   std::size_t{200});
  EXPECT_EQ(cfg.max_trajectory_doublings(),   std::size_t{8});
  EXPECT_EQ(cfg.max_step_halvings(),          std::size_t{3});
  EXPECT_DOUBLE_EQ(cfg.max_hamiltonian_error(), 1.0);
  EXPECT_EQ(cfg.min_micro_steps(),            std::size_t{2});
  EXPECT_DOUBLE_EQ(cfg.rhat_converge_tol(),   1.05);
}

TEST(SamplingConfigBuilder, ChainingReferenceEquality) {
  walnuts::SamplingConfigBuilder builder = walnuts::SamplingConfigBuilder();
  auto& builder_chain = builder
    .min_max_iter(25, 200)
    .max_trajectory_doublings(8)
    .max_step_halvings(3)
    .max_hamiltonian_error(1.0)
    .min_micro_steps(2)
    .rhat_converge_tol(1.05);
  EXPECT_EQ(&builder_chain, &builder);
}

// classes WalnutsConfig and WalnutsConfigBuilder *******************

TEST(WalnutsConfig, MembersAreIndependent) {
  walnuts::WalnutsConfig cfg{
    walnuts::InitConfigBuilder(2, 3).step_sizes(0.25).build(),
      walnuts::WarmupConfigBuilder().min_max_iter(10, 200).build(),
      walnuts::SamplingConfigBuilder().min_max_iter(5, 100).build()
  };

  EXPECT_EQ(cfg.warmup().min_iter(),          std::size_t{10});
  EXPECT_EQ(cfg.warmup().max_iter(),          std::size_t{200});
  EXPECT_EQ(cfg.sampling().min_iter(),        std::size_t{5});
  EXPECT_EQ(cfg.sampling().max_iter(),        std::size_t{100});
  EXPECT_EQ(cfg.init().num_chains(),          std::size_t{2});
  EXPECT_DOUBLE_EQ(cfg.init().step_size(0),   0.25);
  EXPECT_DOUBLE_EQ(cfg.init().step_size(1),   0.25);
}
