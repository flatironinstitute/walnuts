#include <Eigen/Dense>

#include <boost/ut.hpp>

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

/** Mocked nuts::Random that returns a set value deterministically */
struct MockRandom {
  double value;
  double uniform_real_01() { return value; }
};

// Boost.UT allows you to parametrize tests over _types_,
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

// now we can actually start testing

using namespace boost::ut;

void expect_forward_endpoint_equal(auto& span_expected, auto& span_received,
                                   const reflection::source_location& location =
                                       reflection::source_location::current()) {
  expect(span_expected.theta_fw_ == span_received.theta_fw_, location);
  expect(span_expected.rho_fw_ == span_received.rho_fw_, location);
  expect(span_expected.grad_theta_fw_ == span_received.grad_theta_fw_,
         location);
  expect(span_expected.logp_fw_ == span_received.logp_fw_, location);
}

void expect_backward_endpoint_equal(
    auto& span_expected, auto& span_received,
    const reflection::source_location& location =
        reflection::source_location::current()) {
  expect(span_expected.theta_bk_ == span_received.theta_bk_, location);
  expect(span_expected.rho_bk_ == span_received.rho_bk_, location);
  expect(span_expected.grad_theta_bk_ == span_received.grad_theta_bk_,
         location);
  expect(span_expected.logp_bk_ == span_received.logp_bk_, location);
}

void expect_selected_point_equal(auto& span_expected, double logp_expected,
                                 auto& span_received,
                                 const reflection::source_location& location =
                                     reflection::source_location::current()) {
  expect(span_expected.theta_select_ == span_received.theta_select_, location);
  expect(span_expected.grad_select_ == span_received.grad_select_, location);
  expect(approx(logp_expected, span_received.logp_, 1e-6), location);
}

suite<"combine_spans"> tests = [] {
  "endpoints_correct_going_forward"_test = []<typename U> {
    constexpr nuts::Update update = U::value;
    MockRandom rng{0.5};

    nuts::SpanW<double> span_from = dummy_span(0.3);
    nuts::SpanW<double> span_to = dummy_span(0.1);

    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;

    nuts::SpanW<double> combined =
        nuts::combine<update, nuts::Direction::Forward>(
            rng, std::move(span_old), std::move(span_new));

    expect_forward_endpoint_equal(span_to, combined);
    expect_backward_endpoint_equal(span_from, combined);
  } | std::tuple<Metropolis, Barker>();

  "endpoints_correct_going_backward"_test = []<typename U> {
    constexpr nuts::Update update = U::value;
    MockRandom rng{0.5};

    nuts::SpanW<double> span_from = dummy_span(0.3);
    nuts::SpanW<double> span_to = dummy_span(0.1);

    nuts::SpanW<double> span_old = span_from;
    nuts::SpanW<double> span_new = span_to;

    nuts::SpanW<double> combined =
        nuts::combine<update, nuts::Direction::Backward>(
            rng, std::move(span_old), std::move(span_new));

    expect_backward_endpoint_equal(span_to, combined);
    expect_forward_endpoint_equal(span_from, combined);
  } | std::tuple<Metropolis, Barker>();

  "metropolis_accepts_barker_rejects"_test = []<typename D> {
    constexpr nuts::Direction direction = D::value;

    MockRandom rng{0.5};
    nuts::SpanW<double> span_prev = dummy_span(1.2);
    nuts::SpanW<double> span_next = dummy_span(1.0);
    double new_logp = nuts::log_sum_exp(span_prev.logp_, span_next.logp_);

    // log(0.5) = -0.3
    // metropolis: 1.2 - 1.0 = 0.2 > -0.3, accepts
    {
      nuts::SpanW<double> span_old = span_prev;
      nuts::SpanW<double> span_new = span_next;
      nuts::SpanW<double> combined =
          nuts::combine<nuts::Update::Metropolis, direction>(
              rng, std::move(span_old), std::move(span_new));

      expect_selected_point_equal(span_next, new_logp, combined);
    }

    // barker: 1.2 - logsumexp(1.2, 1.0) = 1.2 - 1.78 = -0.58 < -0.3, rejects
    {
      nuts::SpanW<double> span_old = span_prev;
      nuts::SpanW<double> span_new = span_next;
      nuts::SpanW<double> combined =
          nuts::combine<nuts::Update::Barker, direction>(
              rng, std::move(span_old), std::move(span_new));

      expect_selected_point_equal(span_prev, new_logp, combined);
    }
  } | std::tuple<Forward, Backward>();

  // metropolis is always strightly higher than barker, so no case where barker
  // accepts and metropolis rejects for a given random uniform draw

  "barker_and_metropolis_both_reject"_test = []<typename D> {
    constexpr nuts::Direction direction = D::value;

    MockRandom rng{0.5};

    nuts::SpanW<double> span_prev = dummy_span(1.8);
    nuts::SpanW<double> span_next = dummy_span(0.3);
    double new_logp = nuts::log_sum_exp(span_prev.logp_, span_next.logp_);

    // metropolis: 0.3 - 1.8 = -1.5 < -0.3, rejects
    {
      nuts::SpanW<double> span_old = span_prev;
      nuts::SpanW<double> span_new = span_next;
      nuts::SpanW<double> combined =
          nuts::combine<nuts::Update::Metropolis, direction>(
              rng, std::move(span_old), std::move(span_new));

      expect_selected_point_equal(span_prev, new_logp, combined);
    }

    // barker: 0.3 - logsumexp(0.3, 1.8) = 0.3 - 2.00 = -2.00 < -0.3, rejects
    {
      nuts::SpanW<double> span_old = span_prev;
      nuts::SpanW<double> span_new = span_next;
      nuts::SpanW<double> combined =
          nuts::combine<nuts::Update::Barker, direction>(
              rng, std::move(span_old), std::move(span_new));

      expect_selected_point_equal(span_prev, new_logp, combined);
    }
  } | std::tuple<Forward, Backward>();

  "barker_and_metropolis_both_accept"_test = []<typename D> {
    constexpr nuts::Direction direction = D::value;

    MockRandom rng{0.5};
    nuts::SpanW<double> span_prev = dummy_span(0.3);
    nuts::SpanW<double> span_next = dummy_span(1.8);
    double new_logp = nuts::log_sum_exp(span_prev.logp_, span_next.logp_);

    // metropolis: 1.8 - 0.3 = 1.5 > -0.3, accepts
    {
      nuts::SpanW<double> span_old = span_prev;
      nuts::SpanW<double> span_new = span_next;
      nuts::SpanW<double> combined =
          nuts::combine<nuts::Update::Metropolis, direction>(
              rng, std::move(span_old), std::move(span_new));

      expect_selected_point_equal(span_next, new_logp, combined);
    }

    // barker: 1.8 - logsumexp(0.3, 1.8) = 1.8 - 2.0 = -.2 > -0.3, accepts
    {
      nuts::SpanW<double> span_old = span_prev;
      nuts::SpanW<double> span_new = span_next;
      nuts::SpanW<double> combined =
          nuts::combine<nuts::Update::Barker, direction>(
              rng, std::move(span_old), std::move(span_new));

      expect_selected_point_equal(span_next, new_logp, combined);
    }
  } | std::tuple<Forward, Backward>();

  "selected_point_came_from_second_span_on_acceptance"_test =
      []<typename T> {
        constexpr auto direction = std::tuple_element_t<0, T>::value;
        constexpr auto update = std::tuple_element_t<1, T>::value;

        nuts::SpanW<double> span_prev = dummy_span(0.3);
        nuts::SpanW<double> span_next = dummy_span(0.1);
        double new_logp = nuts::log_sum_exp(span_prev.logp_, span_next.logp_);

        MockRandom rng_accept{0};  // always accept

        nuts::SpanW<double> span_old = span_prev;
        nuts::SpanW<double> span_new = span_next;
        nuts::SpanW<double> combined = nuts::combine<update, direction>(
            rng_accept, std::move(span_old), std::move(span_new));

        expect_selected_point_equal(span_next, new_logp, combined);
      } |
      std::tuple<ForwardAndMetropolis, BackwardAndMetropolis, ForwardAndBarker,
                 BackwardAndBarker>();

  "selected_point_came_from_first_span_on_rejection"_test =
      []<typename T> {
        constexpr auto direction = std::tuple_element_t<0, T>::value;
        constexpr auto update = std::tuple_element_t<1, T>::value;

        nuts::SpanW<double> span_prev = dummy_span(0.3);
        nuts::SpanW<double> span_next = dummy_span(0.1);
        double new_logp = nuts::log_sum_exp(span_prev.logp_, span_next.logp_);
        MockRandom rng_reject{1};  // always reject

        nuts::SpanW<double> span_old = span_prev;
        nuts::SpanW<double> span_new = span_next;
        nuts::SpanW<double> combined = nuts::combine<update, direction>(
            rng_reject, std::move(span_old), std::move(span_new));
        expect_selected_point_equal(span_prev, new_logp, combined);
      } |
      std::tuple<ForwardAndMetropolis, BackwardAndMetropolis, ForwardAndBarker,
                 BackwardAndBarker>();
};

}  // namespace span_test
