#include <iostream>

#include "Tensor/Tensor.hpp"

using tensor::Tensor;

int main(){
    std::cout << "--- Tensor.cpp execution started! ---" << std::endl;

    Tensor scalar;
    scalar.set(45.);
    std::cout << scalar.get() << std::endl;
    std::cout << scalar.toString() << std::endl;

    Tensor tensor(3, 3, 3);

    tensor.set(42., 0, 0, 1);
    std::cout << tensor.get(0, 0, 1) << std::endl;

    std::cout << tensor.toString();

    std::cout << "--- Tensor.cpp execution ended! ---" << std::endl;
}
