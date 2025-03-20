#include <iostream>

#include "Tensor/Tensor.hpp"

using tensor::Tensor;

int main(){
    std::cout << "--- Tensor.cpp execution started! ---" << std::endl;

    Tensor tensor(3, 3);

    std::cout << "--- Tensor.cpp execution ended! ---" << std::endl;
}
