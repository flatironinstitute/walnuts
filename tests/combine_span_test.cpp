#include <Eigen/Dense>

#include <gtest/gtest.h>

#include <tuple>
#include <type_traits>
#include <walnuts/walnuts.hpp>

namespace span_test {

/**
 * Returns a span with the specified logp and random values
 * in other fields. The intention is to allow easy comparison
 * on where values in derived spans came from, since two calls
 * to this function will return spans with different forward/backward/selected
 * points.
 */
nuts::SpanW<double> dummy_span(double logp, int size = 2) {
  return nuts::SpanW<double>{Eigen::VectorXd::Random(size),
                             Eigen::VectorXd::Random(size),
                             Eigen::VectorXd::Random(size),
                             -logp,
                             Eigen::VectorXd::Random(size),
                             Eigen::VectorXd::Random(size),
                             Eigen::VectorXd::Random(size),
                             logp + 10,
                             Eigen::VectorXd::Random(size),
                             Eigen::VectorXd::Random(size),
                             logp};
}

/** Mocked nuts::Random that returns a set value deterministically */
struct MockRandom {
  double value;
  double uniform_real_01() { return value; }
};

// Googletest allows you to parametrize tests over _types_,
// but to use non-type template parameters you must first
// wrap them in a type like std::integral_constant
using Forward =
    std::integral_constant<nuts::Direction, nuts::Direction::Forward>;
using Backward =
    std::integral_constant<nuts::Direction, nuts::Direction::Backward>;
using Metropolis =
    std::integral_constant<nuts::Update, nuts::Update::Metropolis>;
using Barker = std::integral_constant<nuts::Update, nuts::Update::Barker>;

using ForwardAndMetropolis = std::tuple<Forward, Metropolis>;
using BackwardAndMetropolis = std::tuple<Backward, Metropolis>;
using ForwardAndBarker = std::tuple<Forward, Barker>;
using BackwardAndBarker = std::tuple<Backward, Barker>;

}  // namespace span_test

// macros to compare parts of Spans
#define EXPECT_FORWARD_ENDPOINT_EQUAL(span_expected, span_received)      \
  EXPECT_EQ(span_expected.theta_fw_, span_received.theta_fw_);           \
  EXPECT_EQ(span_expected.rho_fw_, span_received.rho_fw_);               \
  EXPECT_EQ(span_expected.grad_theta_fw_, span_received.grad_theta_fw_); \
  EXPECT_EQ(span_expected.logp_fw_, span_received.logp_fw_);

#define EXPECT_BACKWARD_ENDPOINT_EQUAL(span_expected, span_received)     \
  EXPECT_EQ(span_expected.theta_bk_, span_received.theta_bk_);           \
  EXPECT_EQ(span_expected.rho_bk_, span_received.rho_bk_);               \
  EXPECT_EQ(span_expected.grad_theta_bk_, span_received.grad_theta_bk_); \
  EXPECT_EQ(span_expected.logp_bk_, span_received.logp_bk_);

#define EXPECT_SELECTED_POINT_EQUAL(span_expected, logp_expected,      \
                                    span_received)                     \
  EXPECT_EQ(span_expected.theta_select_, span_received.theta_select_); \
  EXPECT_EQ(span_expected.grad_select_, span_received.grad_select_);   \
  EXPECT_FLOAT_EQ(logp_expected, span_received.logp_);

// Tests in the CombineSpansUpdateAgnostic suite hold regardless of the update
// rule used (Metropolis or Barker). They can still be sensitive to the
// direction.
template <typename T>
class CombineSpansUpdateAgnostic : public ::testing::Test {};

using BothUpdates = ::testing::Types<span_test::Metropolis, span_test::Barker>;
TYPED_TEST_SUITE(CombineSpansUpdateAgnostic, BothUpdates);

TYPED_TEST(CombineSpansUpdateAgnostic, EndpointsCorrectWhenGoingForward) {
  constexpr auto update = TypeParam::value;

  span_test::MockRandom rng{0};
  nuts::SpanW<double> span_from = span_test::dummy_span(0.3);
  nuts::SpanW<double> span_to = span_test::dummy_span(0.1);

  nuts::SpanW<double> span_old = span_from;
  nuts::SpanW<double> span_new = span_to;
  nuts::SpanW<double> combined =
      nuts::combine<update, nuts::Direction::Forward>(rng, std::move(span_old),
                                                      std::move(span_new));

  EXPECT_BACKWARD_ENDPOINT_EQUAL(span_from, combined);
  EXPECT_FORWARD_ENDPOINT_EQUAL(span_to, combined);
}

TYPED_TEST(CombineSpansUpdateAgnostic, EndpointsCorrectWhenGoingBackward) {
  constexpr auto update = TypeParam::value;

  span_test::MockRandom rng{0};
  nuts::SpanW<double> span_from = span_test::dummy_span(0.3);
  nuts::SpanW<double> span_to = span_test::dummy_span(0.1);

  nuts::SpanW<double> span_old = span_from;
  nuts::SpanW<double> span_new = span_to;
  nuts::SpanW<double> combined =
      nuts::combine<update, nuts::Direction::Backward>(rng, std::move(span_old),
                                                       std::move(span_new));

  EXPECT_BACKWARD_ENDPOINT_EQUAL(span_to, combined);
  EXPECT_FORWARD_ENDPOINT_EQUAL(span_from, combined);
}

// Tests in the CombineSpansSymmetric suite hold regardless of going forwards or
// backwards in time.
template <typename T>
class CombineSpansSymmetric : public ::testing::Test {};

using BothDirections =
    ::testing::Types<span_test::Forward, span_test::Backward>;
TYPED_TEST_SUITE(CombineSpansSymmetric, BothDirections);

TYPED_TEST(CombineSpansSymmetric, MetropolisAcceptsBarkerRejects) {
  constexpr auto direction = TypeParam::value;

  span_test::MockRandom rng{0.5};
  nuts::SpanW<double> span_prev = span_test::dummy_span(1.2);
  nuts::SpanW<double> span_next = span_test::dummy_span(1.0);
  double new_logp = nuts::log_sum_exp(span_prev.logp_, span_next.logp_);

  // log(0.5) = -0.3
  // metropolis: 1.2 - 1.0 = 0.2 > -0.3, accepts
  {
    nuts::SpanW<double> span_old = span_prev;
    nuts::SpanW<double> span_new = span_next;
    nuts::SpanW<double> combined =
        nuts::combine<nuts::Update::Metropolis, direction>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_POINT_EQUAL(span_next, new_logp, combined);
  }

  // barker: 1.2 - logsumexp(1.2, 1.0) = 1.2 - 1.78 = -0.58 < -0.3, rejects
  {
    nuts::SpanW<double> span_old = span_prev;
    nuts::SpanW<double> span_new = span_next;
    nuts::SpanW<double> combined =
        nuts::combine<nuts::Update::Barker, direction>(rng, std::move(span_old),
                                                       std::move(span_new));

    EXPECT_SELECTED_POINT_EQUAL(span_prev, new_logp, combined);
  }
}

// metropolis is always strightly higher than barker, so no case where barker
// accepts and metropolis rejects for a given random uniform draw

TYPED_TEST(CombineSpansSymmetric, BarkerAndMetropolisBothReject) {
  constexpr auto direction = TypeParam::value;

  span_test::MockRandom rng{0.5};

  nuts::SpanW<double> span_prev = span_test::dummy_span(1.8);
  nuts::SpanW<double> span_next = span_test::dummy_span(0.3);
  double new_logp = nuts::log_sum_exp(span_prev.logp_, span_next.logp_);

  // metropolis: 0.3 - 1.8 = -1.5 < -0.3, rejects
  {
    nuts::SpanW<double> span_old = span_prev;
    nuts::SpanW<double> span_new = span_next;
    nuts::SpanW<double> combined =
        nuts::combine<nuts::Update::Metropolis, direction>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_POINT_EQUAL(span_prev, new_logp, combined);
  }

  // barker: 0.3 - logsumexp(0.3, 1.8) = 0.3 - 2.00 = -2.00 < -0.3, rejects
  {
    nuts::SpanW<double> span_old = span_prev;
    nuts::SpanW<double> span_new = span_next;
    nuts::SpanW<double> combined =
        nuts::combine<nuts::Update::Barker, direction>(rng, std::move(span_old),
                                                       std::move(span_new));

    EXPECT_SELECTED_POINT_EQUAL(span_prev, new_logp, combined);
  }
}

TYPED_TEST(CombineSpansSymmetric, BarkerAndMetropolisBothAccept) {
  constexpr auto direction = TypeParam::value;

  span_test::MockRandom rng{0.5};
  nuts::SpanW<double> span_prev = span_test::dummy_span(0.3);
  nuts::SpanW<double> span_next = span_test::dummy_span(1.8);
  double new_logp = nuts::log_sum_exp(span_prev.logp_, span_next.logp_);

  // metropolis: 1.8 - 0.3 = 1.5 > -0.3, accepts
  {
    nuts::SpanW<double> span_old = span_prev;
    nuts::SpanW<double> span_new = span_next;
    nuts::SpanW<double> combined =
        nuts::combine<nuts::Update::Metropolis, direction>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_POINT_EQUAL(span_next, new_logp, combined);
  }

  // barker: 1.8 - logsumexp(0.3, 1.8) = 1.8 - 2.0 = -.2 > -0.3, accepts
  {
    nuts::SpanW<double> span_old = span_prev;
    nuts::SpanW<double> span_new = span_next;
    nuts::SpanW<double> combined =
        nuts::combine<nuts::Update::Barker, direction>(rng, std::move(span_old),
                                                       std::move(span_new));

    EXPECT_SELECTED_POINT_EQUAL(span_next, new_logp, combined);
  }
}

// Tests in the CombineSpansUniversal suite hold regardless of going forwards or
// backwards in time, *and* regardless of the update rule used (Metropolis or
// Barker)
template <typename T>
class CombineSpansUniversal : public ::testing::Test {};

using AllCombinations =
    ::testing::Types<span_test::ForwardAndMetropolis,
                     span_test::BackwardAndMetropolis,
                     span_test::ForwardAndBarker, span_test::BackwardAndBarker>;
TYPED_TEST_SUITE(CombineSpansUniversal, AllCombinations);

TYPED_TEST(CombineSpansUniversal, SelectedPointCameFromSecondSpanOnAcceptance) {
  constexpr auto direction = std::tuple_element_t<0, TypeParam>::value;
  constexpr auto update = std::tuple_element_t<1, TypeParam>::value;

  nuts::SpanW<double> span_prev = span_test::dummy_span(0.3);
  nuts::SpanW<double> span_next = span_test::dummy_span(0.1);
  double new_logp = nuts::log_sum_exp(span_prev.logp_, span_next.logp_);

  span_test::MockRandom rng_accept{0};  // always accept

  nuts::SpanW<double> span_old = span_prev;
  nuts::SpanW<double> span_new = span_next;
  nuts::SpanW<double> combined = nuts::combine<update, direction>(
      rng_accept, std::move(span_old), std::move(span_new));

  EXPECT_SELECTED_POINT_EQUAL(span_next, new_logp, combined);
}

TYPED_TEST(CombineSpansUniversal, SelectedPointCameFromFirstSpanOnRejection) {
  constexpr auto direction = std::tuple_element_t<0, TypeParam>::value;
  constexpr auto update = std::tuple_element_t<1, TypeParam>::value;

  nuts::SpanW<double> span_prev = span_test::dummy_span(0.3);
  nuts::SpanW<double> span_next = span_test::dummy_span(0.1);
  double new_logp = nuts::log_sum_exp(span_prev.logp_, span_next.logp_);

  span_test::MockRandom rng_reject{1};  // always reject

  nuts::SpanW<double> span_old = span_prev;
  nuts::SpanW<double> span_new = span_next;
  nuts::SpanW<double> combined = nuts::combine<update, direction>(
      rng_reject, std::move(span_old), std::move(span_new));

  EXPECT_SELECTED_POINT_EQUAL(span_prev, new_logp, combined);
}

// macro clean up
#undef EXPECT_FORWARD_ENDPOINT_EQUAL
#undef EXPECT_BACKWARD_ENDPOINT_EQUAL
#undef EXPECT_SELECTED_POINT_EQUAL
