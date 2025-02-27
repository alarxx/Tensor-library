#include <iostream>

#include "Tensor/Tensor.hpp"

int main(){
    // try{ } catch(const std::runtime_error& error){
    //     std::cout << error.what() << '\n';
    // }

    std::cout << "--- Tensor.cpp execution started! ---" << std::endl;
    // int arr[0];
    // std::cout << sizeof(arr) << std::endl;

    Tensor tensor(1, 1); // {{0}, {0}}

    Tensor tensor2(1, 1); // {{0}, {0}}
    std::cout << "--- start-move ---" << std::endl;
    tensor = std::move(tensor2); // move assignment operator
    std::cout << "--- end-move ---" << std::endl;

    tensor[0][0] = 1.; // Index Operator [] + move scalar assignment operator
    std::cout << "tensor[0][0]: " << tensor[0][0] << std::endl; // Typecast overloading


    // Tensor vector = {1, 2, 3};
    // Tensor matrix = {
    //     {1, 2, 3},
    //     {4, 5, 6},
    //     {7, 8, 9}
    // };

    std::cout << "--- Tensor.cpp execution ended! ---" << std::endl;

}
