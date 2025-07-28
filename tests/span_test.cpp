

#include <Eigen/Dense>

#include <gtest/gtest.h>

#include <walnuts/walnuts.hpp>

namespace span_test {

/**
  Returns a new span with all the fields filled with randomess except for logp
 */
nuts::SpanW<double> dummy_span(double logp) {
  auto theta = Eigen::VectorXd::Random(2).eval();
  auto rho = Eigen::VectorXd::Random(2).eval();
  auto grad = Eigen::VectorXd::Random(2).eval();

  auto span = nuts::SpanW<double>(std::move(theta), std::move(rho),
                                  std::move(grad), logp);

  span.logp_bk_ = -logp;

  span.theta_bk_ = Eigen::VectorXd::Random(2).eval();
  span.rho_bk_ = Eigen::VectorXd::Random(2).eval();
  span.grad_theta_bk_ = Eigen::VectorXd::Random(2).eval();

  span.logp_fw_ = logp + 10;

  span.theta_fw_ = Eigen::VectorXd::Random(2).eval();
  span.rho_fw_ = Eigen::VectorXd::Random(2).eval();
  span.grad_theta_fw_ = Eigen::VectorXd::Random(2).eval();

  return span;
}

struct mock_rng {
  double value;
  double uniform_real_01() { return value; }
};

}  // namespace span_test

// macros to compare parts of Spans
#define EXPECT_FORWARD_MATCHES(span_recieved, span_expected)             \
  EXPECT_EQ(span_recieved.theta_fw_, span_expected.theta_fw_);           \
  EXPECT_EQ(span_recieved.rho_fw_, span_expected.rho_fw_);               \
  EXPECT_EQ(span_recieved.grad_theta_fw_, span_expected.grad_theta_fw_); \
  EXPECT_EQ(span_recieved.logp_fw_, span_expected.logp_fw_);

#define EXPECT_BACKWARD_MATCHES(span_recieved, span_expected)            \
  EXPECT_EQ(span_recieved.theta_bk_, span_expected.theta_bk_);           \
  EXPECT_EQ(span_recieved.rho_bk_, span_expected.rho_bk_);               \
  EXPECT_EQ(span_recieved.grad_theta_bk_, span_expected.grad_theta_bk_); \
  EXPECT_EQ(span_recieved.logp_bk_, span_expected.logp_bk_);

#define EXPECT_SELECTED_MATCHES(span_recieved, input, new_logp) \
  EXPECT_EQ(span_recieved.theta_select_, input.theta_select_);  \
  EXPECT_EQ(span_recieved.grad_select_, input.grad_select_);    \
  EXPECT_FLOAT_EQ(span_recieved.logp_, new_logp);

TEST(CombineSpans, ForwardBackward) {
  // test that the forward and backward pieces of the new span are as expected

  auto rng = span_test::mock_rng{0};
  auto span_from = span_test::dummy_span(0.3);
  auto span_to = span_test::dummy_span(0.1);

  // forward
  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    auto combined =
        nuts::combine<nuts::Update::Metropolis, nuts::Direction::Forward>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_BACKWARD_MATCHES(combined, span_from);
    EXPECT_FORWARD_MATCHES(combined, span_to);
  }

  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    auto combined =
        nuts::combine<nuts::Update::Barker, nuts::Direction::Forward>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_BACKWARD_MATCHES(combined, span_from);
    EXPECT_FORWARD_MATCHES(combined, span_to);
  }

  // backward
  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    auto combined =
        nuts::combine<nuts::Update::Metropolis, nuts::Direction::Backward>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_BACKWARD_MATCHES(combined, span_to);
    EXPECT_FORWARD_MATCHES(combined, span_from);
  }
  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    auto combined =
        nuts::combine<nuts::Update::Barker, nuts::Direction::Backward>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_BACKWARD_MATCHES(combined, span_to);
    EXPECT_FORWARD_MATCHES(combined, span_from);
  }
}

TEST(CombineSpans, Selected) {
  // test that the 'selected' position and gradient are as expected

  auto rng_accept = span_test::mock_rng{0};  // always accept
  auto span_from = span_test::dummy_span(0.3);
  auto span_to = span_test::dummy_span(0.1);
  auto new_logp = nuts::log_sum_exp(span_from.logp_, span_to.logp_);

  // forward
  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    auto combined =
        nuts::combine<nuts::Update::Metropolis, nuts::Direction::Forward>(
            rng_accept, std::move(span_old), std::move(span_new));

    EXPECT_BACKWARD_MATCHES(combined, span_from);
    EXPECT_FORWARD_MATCHES(combined, span_to);

    EXPECT_SELECTED_MATCHES(combined, span_to, new_logp);
  }

  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    auto combined =
        nuts::combine<nuts::Update::Barker, nuts::Direction::Forward>(
            rng_accept, std::move(span_old), std::move(span_new));

    EXPECT_BACKWARD_MATCHES(combined, span_from);
    EXPECT_FORWARD_MATCHES(combined, span_to);

    EXPECT_SELECTED_MATCHES(combined, span_to, new_logp);
  }

  // backward
  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    auto combined =
        nuts::combine<nuts::Update::Metropolis, nuts::Direction::Backward>(
            rng_accept, std::move(span_old), std::move(span_new));

    EXPECT_BACKWARD_MATCHES(combined, span_to);
    EXPECT_FORWARD_MATCHES(combined, span_from);

    EXPECT_SELECTED_MATCHES(combined, span_to, new_logp);
  }
  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    auto combined =
        nuts::combine<nuts::Update::Barker, nuts::Direction::Backward>(
            rng_accept, std::move(span_old), std::move(span_new));

    EXPECT_BACKWARD_MATCHES(combined, span_to);
    EXPECT_FORWARD_MATCHES(combined, span_from);

    EXPECT_SELECTED_MATCHES(combined, span_to, new_logp);
  }

  // rejecting cases
  auto rng_reject = span_test::mock_rng{1};  // always reject

  // forward
  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    auto combined =
        nuts::combine<nuts::Update::Metropolis, nuts::Direction::Forward>(
            rng_reject, std::move(span_old), std::move(span_new));

    EXPECT_BACKWARD_MATCHES(combined, span_from);
    EXPECT_FORWARD_MATCHES(combined, span_to);

    EXPECT_SELECTED_MATCHES(combined, span_from, new_logp);
  }

  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    auto combined =
        nuts::combine<nuts::Update::Barker, nuts::Direction::Forward>(
            rng_reject, std::move(span_old), std::move(span_new));

    EXPECT_BACKWARD_MATCHES(combined, span_from);
    EXPECT_FORWARD_MATCHES(combined, span_to);

    EXPECT_SELECTED_MATCHES(combined, span_from, new_logp);
  }
  // backward
  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    auto combined =
        nuts::combine<nuts::Update::Metropolis, nuts::Direction::Backward>(
            rng_reject, std::move(span_old), std::move(span_new));

    EXPECT_BACKWARD_MATCHES(combined, span_to);
    EXPECT_FORWARD_MATCHES(combined, span_from);

    EXPECT_SELECTED_MATCHES(combined, span_from, new_logp);
  }
  {
    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;
    auto combined =
        nuts::combine<nuts::Update::Barker, nuts::Direction::Backward>(
            rng_reject, std::move(span_old), std::move(span_new));

    EXPECT_BACKWARD_MATCHES(combined, span_to);
    EXPECT_FORWARD_MATCHES(combined, span_from);

    EXPECT_SELECTED_MATCHES(combined, span_from, new_logp);
  }
}

TEST(CombineSpans, BarkerVsMetropolis) {
  // cases where the two update rules differ

  auto rng = span_test::mock_rng{0.5};
  auto span_from1 = span_test::dummy_span(1.2);
  auto span_to1 = span_test::dummy_span(1.0);
  auto new_logp1 = nuts::log_sum_exp(span_from1.logp_, span_to1.logp_);

  // all tests are symmetric

  // log(0.5) = -0.3
  // metropolis: 1.2 - 1.0 = 0.2 > -0.3, accepts
  {
    nuts::SpanW<double> span_old = span_from1;
    nuts::SpanW<double> span_new = span_to1;
    auto combined =
        nuts::combine<nuts::Update::Metropolis, nuts::Direction::Forward>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_MATCHES(combined, span_to1, new_logp1);
  }
  {
    nuts::SpanW<double> span_old = span_from1;
    nuts::SpanW<double> span_new = span_to1;
    auto combined =
        nuts::combine<nuts::Update::Metropolis, nuts::Direction::Backward>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_MATCHES(combined, span_to1, new_logp1);
  }

  // barker: 1.2 - lsexp(1.2, 1.0) = 1.2 - 1.78 = -0.58 < -0.3, rejects
  {
    nuts::SpanW<double> span_old = span_from1;
    nuts::SpanW<double> span_new = span_to1;
    auto combined =
        nuts::combine<nuts::Update::Barker, nuts::Direction::Forward>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_MATCHES(combined, span_from1, new_logp1);
  }
  {
    nuts::SpanW<double> span_old = span_from1;
    nuts::SpanW<double> span_new = span_to1;
    auto combined =
        nuts::combine<nuts::Update::Barker, nuts::Direction::Backward>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_MATCHES(combined, span_from1, new_logp1);
  }

  auto span_from2 = span_test::dummy_span(1.8);
  auto span_to2 = span_test::dummy_span(0.3);
  auto new_logp2 = nuts::log_sum_exp(span_from2.logp_, span_to2.logp_);

  // metropolis: 0.3 - 1.8 = -1.5 < -0.3, rejects
  {
    nuts::SpanW<double> span_old = span_from2;
    nuts::SpanW<double> span_new = span_to2;
    auto combined =
        nuts::combine<nuts::Update::Metropolis, nuts::Direction::Forward>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_MATCHES(combined, span_from2, new_logp2);
  }
  {
    nuts::SpanW<double> span_old = span_from2;
    nuts::SpanW<double> span_new = span_to2;
    auto combined =
        nuts::combine<nuts::Update::Metropolis, nuts::Direction::Backward>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_MATCHES(combined, span_from2, new_logp2);
  }

  // barker: 0.3 - lsexp(0.3, 1.8) = 0.3 - 2.00 = -2.00 < -0.3, rejects
  {
    nuts::SpanW<double> span_old = span_from2;
    nuts::SpanW<double> span_new = span_to2;
    auto combined =
        nuts::combine<nuts::Update::Barker, nuts::Direction::Forward>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_MATCHES(combined, span_from2, new_logp2);
  }
  {
    nuts::SpanW<double> span_old = span_from2;
    nuts::SpanW<double> span_new = span_to2;
    auto combined =
        nuts::combine<nuts::Update::Barker, nuts::Direction::Backward>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_MATCHES(combined, span_from2, new_logp2);
  }

  auto span_from3 = span_test::dummy_span(0.3);
  auto span_to3 = span_test::dummy_span(1.8);
  auto new_logp3 = nuts::log_sum_exp(span_from3.logp_, span_to3.logp_);

  // metropolis: 1.8 - 0.3 = 1.5 > -0.3, accepts
  {
    nuts::SpanW<double> span_old = span_from3;
    nuts::SpanW<double> span_new = span_to3;
    auto combined =
        nuts::combine<nuts::Update::Metropolis, nuts::Direction::Forward>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_MATCHES(combined, span_to3, new_logp3);
  }
  {
    nuts::SpanW<double> span_old = span_from3;
    nuts::SpanW<double> span_new = span_to3;
    auto combined =
        nuts::combine<nuts::Update::Metropolis, nuts::Direction::Backward>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_MATCHES(combined, span_to3, new_logp3);
  }

  // barker: 1.8 - lsexp(0.3, 1.8) = 1.8 - 2.0 = -.2 > -0.3, accepts
  {
    nuts::SpanW<double> span_old = span_from3;
    nuts::SpanW<double> span_new = span_to3;
    auto combined =
        nuts::combine<nuts::Update::Barker, nuts::Direction::Forward>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_MATCHES(combined, span_to3, new_logp3);
  }
  {
    nuts::SpanW<double> span_old = span_from3;
    nuts::SpanW<double> span_new = span_to3;
    auto combined =
        nuts::combine<nuts::Update::Barker, nuts::Direction::Backward>(
            rng, std::move(span_old), std::move(span_new));

    EXPECT_SELECTED_MATCHES(combined, span_to3, new_logp3);
  }

  // metropolis is always strightly higher than barker, so no case where barker
  // accepts and metropolis rejects for a given random uniform draw
}
// macro clean up
#undef EXPECT_FORWARD_MATCHES
#undef EXPECT_BACKWARD_MATCHES
#undef EXPECT_SELECTED_MATCHES
