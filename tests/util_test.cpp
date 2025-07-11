#include <cmath>

#include <Eigen/Dense>

#include <gtest/gtest.h>

#include <walnuts/util.hpp>
#include <walnuts/walnuts.hpp>
#include <walnuts/nuts.hpp>

using S = double;
using Vec = Eigen::Matrix<S, -1, 1>;
using Mat = Eigen::Matrix<S, -1, -1>;

static Vec vec(S x1, S x2) {
  Vec y(2);
  y << x1, x2;
  return y;
}

static Mat mat(S x00, S x01, S x10, S x11) {
  Mat y(2, 2);
  y << x00, x01, x10, x11;
  return y;
}

TEST(Util, Dummy) {
  EXPECT_EQ(1 + 1, 2);
}

TEST(Util, Walnuts) {
  EXPECT_EQ(2 + 2, 4);
  Vec thetabk1 = vec(-3, 0);
  Vec thetafw1 = vec(-1, 0);
  Vec thetabk2 = vec(1, 0);
  Vec thetafw2 = vec(3, 0);

  Vec rhobk1 = vec(1, -1);
  Vec rhofw1 = vec(0, 1);
  Vec rhobk2 = vec(0, 1);
  Vec rhofw2 = vec(-1, -1);

  // unused for U-turn, but needed for span
  Vec gradbk1 = vec(0, 0);
  Vec gradfw1 = vec(0, 0);
  Vec gradbk2 = vec(0, 0);
  Vec gradfw2 = vec(0, 0);

  S logpbk1 = 0;
  S logpfw1 = 0;
  S logpbk2 = 0;
  S logpfw2 = 0;

  Vec theta1 = vec(0, 0);
  Vec theta2 = vec(0, 0);
  Vec grad1 = vec(0, 0);
  Vec grad2 = vec(0, 0);
  
  S logp1 = 0;
  S logp2 = 0;

  Mat inv_mass = mat(1, 0, 0, 1);

  nuts::SpanW<S> span1bk(std::move(thetabk1), std::move(rhobk1), std::move(gradbk1), logpbk1);
  nuts::SpanW<S> span1fw(std::move(thetafw1), std::move(rhofw1), std::move(gradfw1), logpfw1);
  nuts::SpanW<S> span2bk(std::move(thetabk2), std::move(rhobk2), std::move(gradbk2), logpbk2);
  nuts::SpanW<S> span2fw(std::move(thetafw2), std::move(rhofw2), std::move(gradfw2), logpfw2);
  
  nuts::SpanW<S> span1(std::move(span1bk), std::move(span1fw), std::move(theta1), std::move(grad1), logp1);
  nuts::SpanW<S> span2(std::move(span2bk), std::move(span2fw), std::move(theta2), std::move(grad2), logp2);

  EXPECT_TRUE((nuts::uturn<nuts::Direction::Forward, S, nuts::SpanW<S>>(span1, span2, inv_mass)));
  EXPECT_FALSE((nuts::uturn<nuts::Direction::Forward, S, nuts::SpanW<S>>(span2, span1, inv_mass)));
}


TEST(Util, WalnutsRegression) {
  EXPECT_EQ(2 + 2, 4);
  Vec thetabk1 = vec(3, 0);
  Vec thetafw1 = vec(0, 0);
  Vec thetabk2 = vec(1, 0);
  Vec thetafw2 = vec(3, 0);

  Vec rhobk1 = vec(-1, 1);
  Vec rhofw1 = vec(0, 1);
  Vec rhobk2 = vec(0, 1);
  Vec rhofw2 = vec(1, -1);

  // unused for U-turn, but needed for span
  Vec gradbk1 = vec(0, 0);
  Vec gradfw1 = vec(0, 0);
  Vec gradbk2 = vec(0, 0);
  Vec gradfw2 = vec(0, 0);

  S logpbk1 = 0;
  S logpfw1 = 0;
  S logpbk2 = 0;
  S logpfw2 = 0;

  Vec theta1 = vec(0, 0);
  Vec theta2 = vec(0, 0);
  Vec grad1 = vec(0, 0);
  Vec grad2 = vec(0, 0);
  
  S logp1 = 0;
  S logp2 = 0;

  Mat inv_mass = mat(1, 0, 0, 1);

  nuts::SpanW<S> span1bk(std::move(thetabk1), std::move(rhobk1), std::move(gradbk1), logpbk1);
  nuts::SpanW<S> span1fw(std::move(thetafw1), std::move(rhofw1), std::move(gradfw1), logpfw1);
  nuts::SpanW<S> span2bk(std::move(thetabk2), std::move(rhobk2), std::move(gradbk2), logpbk2);
  nuts::SpanW<S> span2fw(std::move(thetafw2), std::move(rhofw2), std::move(gradfw2), logpfw2);
  
  nuts::SpanW<S> span1(std::move(span1bk), std::move(span1fw), std::move(theta1), std::move(grad1), logp1);
  nuts::SpanW<S> span2(std::move(span2bk), std::move(span2fw), std::move(theta2), std::move(grad2), logp2);

  // following test fails in the original code with buggy uturn condition
  EXPECT_FALSE((nuts::uturn<nuts::Direction::Forward, S, nuts::SpanW<S>>(span1, span2, inv_mass)));
}
