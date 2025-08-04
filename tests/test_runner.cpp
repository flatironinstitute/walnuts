#include <boost/ut.hpp>

int main(int argc, const char* argv[]) {
  namespace ut = boost::ut;
  return ut::cfg<ut::override>.run({.argc = argc, .argv = argv});
}
