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

// int main(){
//     std::cout << "--- Tensor.cpp execution started! ---" << std::endl;
//
//
//     Tensor tensor = {1, 2, 3};
//
//     std::cout << "\nCopy constructor example:" << std::endl;
//     // Tensor<double> copy_a = {1., 1., 1.}; // Error on copy assignment
//     Tensor copy_c = tensor;
//
//     copy_c[0] = 40;
//
//     copy_c.print();
//     tensor.print();
//
//     std::cout << "\nCopy assignment example:" << std::endl;
//     // Tensor<double> copy_a = {1., 1., 1.}; // Error on copy assignment
//     Tensor copy_a = {1, 1, 1};
//     copy_a = tensor;
//
//     copy_a[0] = 42;
//
//     copy_a.print();
//     tensor.print();
//
//     std::cout << "\n--- Tensor.cpp execution ended! ---" << std::endl;
// }

// int main(){
//     std::cout << "--- Tensor.cpp execution started! ---" << std::endl;
//
//     Tensor t1 = {1, 2};
//     Tensor t2 = {3, 4};
//
//     std::cout << "\nAppending tensors example:" << std::endl;
//
//     // Tensor tensor = {t1, t2}; // copy
//     // Tensor tensor = {Tensor({1, 2}), Tensor({3, 4})}; // rvalue - move
//     Tensor tensor = { std::move(t1), std::move(t2) }; // move, лучше всегда делать так
//
//     t1.print();
//     std::cout << std::endl;
//     t2.print();
//     std::cout << std::endl;
//     tensor.print();
//     /*
//     Output:
//         tensor<i>:
//         0i
//         tensor<i>:
//         0i
//         tensor<i>:
//         1i 2i
//         3i 4i
//     */
//
//     std::cout << "\n--- Tensor.cpp execution ended! ---" << std::endl;
// }

// int main(){
//     std::cout << "--- Tensor.cpp execution started! ---" << std::endl;
//
//     Tensor tensor = {
//         {1, 2, 3},
//         {4, 5, 6},
//         {7, 8, 9}
//     };
//     Tensor copy = tensor;
//
//     std::cout << tensor;
//
//     std::cout << "\n--- Tensor.cpp execution ended! ---" << std::endl;
// }

int main(){
    std::cout << "--- Tensor.cpp execution started! ---" << std::endl;

    // Tensor vec(3); // double vector
    // std::cout << vec << std::endl;
    // Tensor sc = scalar(1.); // double scalar
    // std::cout << sc << std::endl;

    // Tensor t1 = {1, 2, 3};
    // Tensor t2 = {4, 5, 6};
    Tensor t1 = {
        {1, 2, 3},
        {4, 5, 6}
    };
    Tensor t2 = {
        {6, 5, 4},
        {3, 2, 1}
    };
    std::cout << t1 << std::endl;
    std::cout << t2 << std::endl;

    // std::cout << "\nUnary multiplication example:" << std::endl;
    // t1 *= t2;
    // std::cout << std::endl;
    // std::cout << t1 << std::endl;
    // std::cout << t2 << std::endl;

    std::cout << "\nBinary multiplication example:" << std::endl;
    Tensor tensor = t1 * t2; // without RVO copying could be 2 times
    std::cout << t1 << std::endl;
    std::cout << t2 << std::endl;
    std::cout << tensor;

    std::cout << "\n--- Tensor.cpp execution ended! ---" << std::endl;
}

/*
- [x] Нужно добавить copy конструктор и assignment operator ?

- [x] Потому что я хочу красивый синтаксис Appending-а тензоров
Да и в будущем это понадобится, потому что copy метод может делать 2 раза копию без RVO
- [ ] А appending функцию по идее без копирования можно сделать через TArgs... и move, но он без RVO снова будет делать копию? Я не знаю, по идее...

- [ ] Нужно как-то добавить casting между Tensor<double> и Tensor<int> например.

- [x] toString

- [ ] += operator
    - [ ] traversing till vector or matrix
- [ ] + operator
- [ ] no need in namespace
- [ ] iterator

*/
