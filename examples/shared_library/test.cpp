#include <iostream>

extern "C" void test() {
  std::cout << "Haha" << std::endl;
  std::cout << "It works!" << std::endl;
}