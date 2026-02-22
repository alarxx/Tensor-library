#include <iostream>

#include <TensorX/Config.h>


void print_cxx_std(){
// https://stackoverflow.com/questions/2324658/how-to-determine-the-version-of-the-c-standard-used-by-the-compiler
  if (__cplusplus == 202101L) std::cout << "C++23";
  else if (__cplusplus == 202002L) std::cout << "C++20";
  else if (__cplusplus == 201703L) std::cout << "C++17";
  else if (__cplusplus == 201402L) std::cout << "C++14";
  else if (__cplusplus == 201103L) std::cout << "C++11";
  else if (__cplusplus == 199711L) std::cout << "C++98";
  else std::cout << "pre-standard C++." << __cplusplus;
  std::cout << "\n";
}


int main(int argc, char* argv[]){
    print_cxx_std();

    #ifdef USE_EXAMPLES
        std::cout << "USE_EXAMPLES is defined" << std::endl;
    #endif

    std::cout << "Config.h.in" << std::endl;
    std::cout << "\t MY_CUSTOM_VARIABLE: " << MY_CUSTOM_VARIABLE << std::endl;
    std::cout << "\t Version: " << TensorX_VERSION_MAJOR << "." << TensorX_VERSION_MINOR << std::endl;
}
