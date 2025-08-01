#include <boost/ut.hpp>
#include <cstdlib>
#include <iostream>

// a small custom runner for boost.UT tests
int main(int argc, const char* argv[]) {
  // global settings we want for all tests
  boost::ut::detail::cfg::show_successful_tests = true;
  boost::ut::detail::cfg::show_duration = true;

  if (argc == 2) {
    std::string arg(argv[1]);
    if (arg == "--list") {
      // tell runner to print names
      boost::ut::detail::cfg::show_test_names = true;
      // and to _not_ do anything else
      boost::ut::cfg<boost::ut::override> = {.dry_run = true};
    } else if (arg.starts_with("-")) {
      // catch-all case for both --help and any other unknown options
      if (arg != "--help") {
        std::cout << "Unknown option: " << arg << std::endl;
      }
      std::cout << "Usage: " << argv[0] << " <options> [filter]" << std::endl;
      std::cout << "Run all tests or filter by a specific test name."
                << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  --help       Show this help message" << std::endl;
      std::cout << "  --list       List all available tests" << std::endl;
      // _Exit avoids running static destructors, which is what triggers
      // tests to run (if they haven't already)
      std::_Exit(1);
    } else {
      // any other string we treat as a filter
      boost::ut::cfg<boost::ut::override> = {.filter = argv[1]};
    }
  }
}
