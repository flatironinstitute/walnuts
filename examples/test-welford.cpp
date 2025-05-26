#include <iostream>
#include <walnuts/welford.hpp>

using S = double;
using Vec = Eigen::Matrix<S, Eigen::Dynamic, 1>;
using walnuts::DiscountedOnlineMoments;

void report(const std::string& label, const DiscountedOnlineMoments<S>& dom) {
  std::cout << label << std::endl;
  std::cout << "Mean: " << dom.mean().transpose() << std::endl;
  std::cout << "Variance: " << dom.variance().transpose().eval() << std::endl;
}

int main() {
  constexpr S alpha = 0.9;
  constexpr int dim = 2;
  DiscountedOnlineMoments<S> dom(alpha, dim);
  Vec x(dim);

  report("\nZero observations:", dom);

  x << 1.0, 2.0;
  dom.update(x);
  report("\nOne observation y[1] = [1 2]:", dom);

  for (S i = 1; i <= 10; ++i) {
    x << i, 2 * i;
    dom.update(x);
  }
  report("\nTen observations ([1 2], [2, 4], ..., [10 20]):", dom);

  return 0;
}
