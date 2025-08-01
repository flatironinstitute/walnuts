#include <boost/ut.hpp>

// a small custom runner for boost.UT tests
int main(int argc, const char* argv[]) {
  boost::ut::detail::cfg::show_successful_tests = true;
  boost::ut::detail::cfg::show_duration = true;

  if (argc == 2) {
    if (std::string(argv[1]) == "--help") {
      std::cout << "Usage: " << argv[0] << " <options> [filter]\n";
      std::cout << "Run all tests or filter by a specific test name.\n";
      std::cout << "Options:\n";
      std::cout << "  --help       Show this help message\n";
      std::cout << "  --list       List all available tests\n";
      // _Exit avoids running static destructors, which is what triggers
      // tests to run (if they haven't already)
      _Exit(1);
    } else if (std::string(argv[1]) == "--list") {
      // tell runner to print names
      boost::ut::detail::cfg::show_test_names = true;
      // an to _not_ do anything else
      boost::ut::cfg<boost::ut::override> = {.dry_run = true};
    } else {
      boost::ut::cfg<boost::ut::override> = {.filter = argv[1]};
    }
  }
}
