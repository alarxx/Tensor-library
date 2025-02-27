#include <iostream>

#include "Tensor/Tensor.hpp"

// int main(){
//     // try{ } catch(const std::runtime_error& error){
//     //     std::cout << error.what() << '\n';
//     // }
//
//     std::cout << "--- Tensor.cpp execution started! ---" << std::endl;
//     // int arr[0];
//     // std::cout << sizeof(arr) << std::endl;
//
//     Tensor tensor(1, 1); // {{0}, {0}}
//
//     Tensor tensor2(1, 1); // {{0}, {0}}
//     std::cout << "--- start-move ---" << std::endl;
//     tensor = std::move(tensor2); // move assignment operator
//     std::cout << "--- end-move ---" << std::endl;
//
//     tensor[0][0] = 1.; // Index Operator [] + move scalar assignment operator
//     std::cout << "tensor[0][0]: " << tensor[0][0] << std::endl; // Typecast overloading
//
//     std::cout << "--- Tensor.cpp execution ended! ---" << std::endl;
// }

// int main(){
//     std::cout << "--- Tensor.cpp execution started! ---" << std::endl;
//
//
//     std::cout << "\nVector example:" << std::endl;
//     // Vector <int> (deduction)
//     Tensor vector = {1, 2, 3};
//
//     std::cout << "vector<" << typeid(decltype(vector)::type).name() << ">:" << std::endl;
//     for(int i = 0; i < vector.size(); i++){
//         std::cout << vector[i] << typeid(vector[i].value()).name() << " ";
//     }
//     std::cout << std::endl;
//
//
//     std::cout << "\nMatrix example:" << std::endl;
//     // Matrix <double>
//     Tensor<double> matrix = {
//         {1, 2},
//         {4, 5}, // {4, 5, 6}, // we can do like this, actually
//     };
//
//     std::cout << "matrix<" << typeid(decltype(matrix)::type).name() << ">:" << std::endl;
//     for(int i = 0; i < matrix.size(); i++){
//         for(int j = 0; j < matrix[i].size(); j++){
//             std::cout << matrix[i][j] << typeid(matrix[i][j].value()).name() << " ";
//         }
//         std::cout << std::endl;
//     }
//
//
//     std::cout << "\n--- Tensor.cpp execution ended! ---" << std::endl;
// }

int main(){
    std::cout << "--- Tensor.cpp execution started! ---" << std::endl;


    Tensor tensor = {1, 2, 3};

    std::cout << "\nCopy constructor example:" << std::endl;
    // Tensor<double> copy_a = {1., 1., 1.}; // Error on copy assignment
    Tensor copy_c = tensor;

    copy_c[0] = 40;

    copy_c.print();
    tensor.print();

    std::cout << "\nCopy assignment example:" << std::endl;
    // Tensor<double> copy_a = {1., 1., 1.}; // Error on copy assignment
    Tensor copy_a = {1, 1, 1};
    copy_a = tensor;

    copy_a[0] = 42;

    copy_a.print();
    tensor.print();

    std::cout << "\n--- Tensor.cpp execution ended! ---" << std::endl;
}
