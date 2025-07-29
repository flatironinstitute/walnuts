#include <Eigen/Dense>

#include <gtest/gtest.h>

#include <tuple>
#include <type_traits>
#include <walnuts/walnuts.hpp>

namespace span_test {

/**
  Returns a new span with all the fields filled with randomess except for
  logp
 */
nuts::SpanW<double> dummy_span(double logp) {
  Eigen::VectorXd theta = Eigen::VectorXd::Random(2);
  Eigen::VectorXd rho = Eigen::VectorXd::Random(2);
  Eigen::VectorXd grad = Eigen::VectorXd::Random(2);

  nuts::SpanW<double> span(std::move(theta), std::move(rho), std::move(grad),
                           logp);

  span.logp_bk_ = -logp;

  span.theta_bk_ = Eigen::VectorXd::Random(2);
  span.rho_bk_ = Eigen::VectorXd::Random(2);
  span.grad_theta_bk_ = Eigen::VectorXd::Random(2);

  span.logp_fw_ = logp + 10;

  span.theta_fw_ = Eigen::VectorXd::Random(2);
  span.rho_fw_ = Eigen::VectorXd::Random(2);
  span.grad_theta_fw_ = Eigen::VectorXd::Random(2);

  return span;
}

/**  Mocked RNG that returns a set value deterministically */
struct mock_rng {
  double value;
  double uniform_real_01() { return value; }
};

// helpers to use GoogleTest's parameterization system
using Forwards =
    std::integral_constant<nuts::Direction, nuts::Direction::Forward>;
using Backwards =
    std::integral_constant<nuts::Direction, nuts::Direction::Backward>;
using Metropolis =
    std::integral_constant<nuts::Update, nuts::Update::Metropolis>;
using Barker = std::integral_constant<nuts::Update, nuts::Update::Barker>;

using ForwardsMetropolis = std::tuple<Forwards, Metropolis>;
using BackwardsMetropolis = std::tuple<Backwards, Metropolis>;
using ForwardsBarker = std::tuple<Forwards, Barker>;
using BackwardsBarker = std::tuple<Backwards, Barker>;

}  // namespace span_test

// macros to compare parts of Spans
#define EXPECT_FORWARD_ENDPOINT_EQUAL(span_recieved, span_expected)      \
  EXPECT_EQ(span_recieved.theta_fw_, span_expected.theta_fw_);           \
  EXPECT_EQ(span_recieved.rho_fw_, span_expected.rho_fw_);               \
  EXPECT_EQ(span_recieved.grad_theta_fw_, span_expected.grad_theta_fw_); \
  EXPECT_EQ(span_recieved.logp_fw_, span_expected.logp_fw_);

#define EXPECT_BACKWARD_ENDPOINT_EQUAL(span_recieved, span_expected)     \
  EXPECT_EQ(span_recieved.theta_bk_, span_expected.theta_bk_);           \
  EXPECT_EQ(span_recieved.rho_bk_, span_expected.rho_bk_);               \
  EXPECT_EQ(span_recieved.grad_theta_bk_, span_expected.grad_theta_bk_); \
  EXPECT_EQ(span_recieved.logp_bk_, span_expected.logp_bk_);

#define EXPECT_SELECTED_POINT_EQUAL(span_recieved, input, new_logp) \
  EXPECT_EQ(span_recieved.theta_select_, input.theta_select_);      \
  EXPECT_EQ(span_recieved.grad_select_, input.grad_select_);        \
  EXPECT_FLOAT_EQ(span_recieved.logp_, new_logp);

// The forwards/backwards pieces of the new span are update agnostic
template <typename T>
class CombineSpansUpdateAgnostic : public ::testing::Test {};

using BothUpdates = ::testing::Types<span_test::Metropolis, span_test::Barker>;
TYPED_TEST_SUITE(CombineSpansUpdateAgnostic, BothUpdates);

TYPED_TEST(CombineSpansUpdateAgnostic, ForwardEndpointsCorrect) {
  constexpr auto update = TypeParam::value;

  span_test::mock_rng rng{0};
  nuts::SpanW<double> span_from = span_test::dummy_span(0.3);
  nuts::SpanW<double> span_to = span_test::dummy_span(0.1);

  nuts::SpanW<double> span_old = span_from;
  nuts::SpanW<double> span_new = span_to;
  nuts::SpanW<double> combined =
      nuts::combine<update, nuts::Direction::Forward>(rng, std::move(span_old),
                                                      std::move(span_new));

  EXPECT_BACKWARD_ENDPOINT_EQUAL(combined, span_from);
  EXPECT_FORWARD_ENDPOINT_EQUAL(combined, span_to);
}

TYPED_TEST(CombineSpansUpdateAgnostic, BackwardsEndpointsCorrect) {
  constexpr auto update = TypeParam::value;

  span_test::mock_rng rng{0};
  nuts::SpanW<double> span_from = span_test::dummy_span(0.3);
  nuts::SpanW<double> span_to = span_test::dummy_span(0.1);

  nuts::SpanW<double> span_old = span_from;
  nuts::SpanW<double> span_new = span_to;
  nuts::SpanW<double> combined =
      nuts::combine<update, nuts::Direction::Backward>(rng, std::move(span_old),
                                                       std::move(span_new));

  EXPECT_BACKWARD_ENDPOINT_EQUAL(combined, span_to);
  EXPECT_FORWARD_ENDPOINT_EQUAL(combined, span_from);
}

// following tests are symmetric (same whether going forwards or backwards)
template <typename T>
class CombineSpansSymmetric : public ::testing::Test {};

using BothDirections =
    ::testing::Types<span_test::Forwards, span_test::Backwards>;
TYPED_TEST_SUITE(CombineSpansSymmetric, BothDirections);

TYPED_TEST(CombineSpansSymmetric, MetropolisAcceptsBarkerRejects) {
  // cases where the two update rules differ
  constexpr auto direction = TypeParam::value;

  span_test::mock_rng rng{0.5};
  nuts::SpanW<double> span_from = span_test::dummy_span(1.2);
  nuts::SpanW<double> span_to = span_test::dummy_span(1.0);
  double new_logp = nuts::log_sum_exp(span_from.logp_, span_to.logp_);

  // log(0.5) = -0.3
  // metropolis: 1.2 - 1.0 = 0.2 > -0.3, accepts
  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    nuts::SpanW<double> combined =
        nuts::combine<nuts::Update::Metropolis, direction>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_POINT_EQUAL(combined, span_to, new_logp);
  }

  // barker: 1.2 - lsexp(1.2, 1.0) = 1.2 - 1.78 = -0.58 < -0.3, rejects
  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    nuts::SpanW<double> combined =
        nuts::combine<nuts::Update::Barker, direction>(rng, std::move(span_old),
                                                       std::move(span_new));

    EXPECT_SELECTED_POINT_EQUAL(combined, span_from, new_logp);
  }
}

// metropolis is always strightly higher than barker, so no case where barker
// accepts and metropolis rejects for a given random uniform draw

TYPED_TEST(CombineSpansSymmetric, BarkerAndMetropolisBothReject) {
  // cases where the two update rules differ
  constexpr auto direction = TypeParam::value;

  span_test::mock_rng rng{0.5};

  nuts::SpanW<double> span_from = span_test::dummy_span(1.8);
  nuts::SpanW<double> span_to = span_test::dummy_span(0.3);
  double new_logp = nuts::log_sum_exp(span_from.logp_, span_to.logp_);

  // metropolis: 0.3 - 1.8 = -1.5 < -0.3, rejects
  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    nuts::SpanW<double> combined =
        nuts::combine<nuts::Update::Metropolis, direction>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_POINT_EQUAL(combined, span_from, new_logp);
  }

  // barker: 0.3 - lsexp(0.3, 1.8) = 0.3 - 2.00 = -2.00 < -0.3, rejects
  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    nuts::SpanW<double> combined =
        nuts::combine<nuts::Update::Barker, direction>(rng, std::move(span_old),
                                                       std::move(span_new));

    EXPECT_SELECTED_POINT_EQUAL(combined, span_from, new_logp);
  }
}

TYPED_TEST(CombineSpansSymmetric, BarkerAndMetropolisBothAccept) {
  // cases where the two update rules differ
  constexpr auto direction = TypeParam::value;

  span_test::mock_rng rng{0.5};
  nuts::SpanW<double> span_from = span_test::dummy_span(0.3);
  nuts::SpanW<double> span_to = span_test::dummy_span(1.8);
  double new_logp = nuts::log_sum_exp(span_from.logp_, span_to.logp_);

  // metropolis: 1.8 - 0.3 = 1.5 > -0.3, accepts
  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    nuts::SpanW<double> combined =
        nuts::combine<nuts::Update::Metropolis, direction>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_POINT_EQUAL(combined, span_to, new_logp);
  }

  // barker: 1.8 - lsexp(0.3, 1.8) = 1.8 - 2.0 = -.2 > -0.3, accepts
  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    nuts::SpanW<double> combined =
        nuts::combine<nuts::Update::Barker, direction>(rng, std::move(span_old),
                                                       std::move(span_new));

    EXPECT_SELECTED_POINT_EQUAL(combined, span_to, new_logp);
  }
}

// The following test uses the same logic for all combinations of
// direction and update type
template <typename T>
class CombineSpansUniversal : public ::testing::Test {};

using AllCombinations =
    ::testing::Types<span_test::ForwardsMetropolis,
                     span_test::BackwardsMetropolis, span_test::ForwardsBarker,
                     span_test::BackwardsBarker>;
TYPED_TEST_SUITE(CombineSpansUniversal, AllCombinations);

TYPED_TEST(CombineSpansUniversal, SelectedPositionAccepted) {
  // test that the 'selected' position and gradient are as expected
  constexpr auto direction = std::tuple_element_t<0, TypeParam>::value;
  constexpr auto update = std::tuple_element_t<1, TypeParam>::value;

  nuts::SpanW<double> span_from = span_test::dummy_span(0.3);
  nuts::SpanW<double> span_to = span_test::dummy_span(0.1);
  double new_logp = nuts::log_sum_exp(span_from.logp_, span_to.logp_);

  span_test::mock_rng rng_accept{0};  // always accept

  nuts::SpanW<double> span_old = span_from;
  nuts::SpanW<double> span_new = span_to;
  nuts::SpanW<double> combined = nuts::combine<update, direction>(
      rng_accept, std::move(span_old), std::move(span_new));

  EXPECT_SELECTED_POINT_EQUAL(combined, span_to, new_logp);
}

TYPED_TEST(CombineSpansUniversal, SelectedPositionRejected) {
  // test that the 'selected' position and gradient are as expected
  constexpr auto direction = std::tuple_element_t<0, TypeParam>::value;
  constexpr auto update = std::tuple_element_t<1, TypeParam>::value;

  nuts::SpanW<double> span_from = span_test::dummy_span(0.3);
  nuts::SpanW<double> span_to = span_test::dummy_span(0.1);
  double new_logp = nuts::log_sum_exp(span_from.logp_, span_to.logp_);

  span_test::mock_rng rng_reject{1};  // always reject

  nuts::SpanW<double> span_old = span_from;
  nuts::SpanW<double> span_new = span_to;
  nuts::SpanW<double> combined = nuts::combine<update, direction>(
      rng_reject, std::move(span_old), std::move(span_new));

  EXPECT_SELECTED_POINT_EQUAL(combined, span_from, new_logp);
}

// macro clean up
#undef EXPECT_FORWARD_ENDPOINT_EQUAL
#undef EXPECT_BACKWARD_ENDPOINT_EQUAL
#undef EXPECT_SELECTED_POINT_EQUAL
