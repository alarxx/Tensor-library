/*
    SPDX-License-Identifier: MPL-2.0
    --------------------------------
    This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
    If a copy of the MPL was not distributed with this file,
    You can obtain one at https://mozilla.org/MPL/2.0/.

    This file is part of the Tensor-library:
    https://github.com/alarxx/Tensor-library

    Provided “as is”, without warranty of any kind.

    Copyright © 2025 Alar Akilbekov. All rights reserved.
 */

#include <iostream>
#include <type_traits>
#include <concepts>
#include <initializer_list>

#include "Tensor.hpp"

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>; // C++20

/*
    Я делаю Tensor homogeneous, то есть он хранит данные только одного типа.

    Можно реализовать Double-Tensor,
    но здесь реализовано лучшее решение сделать template Tensor,
    где user может выбирать тип Tensor-а, по умолчанию double.

        template <typename T = double>
        class Tensor {

    Tensor проводит арифмитические операции сложения, вычитания, умножения, деления и т.д.,
    поэтому мы должны ограничить возможные типы только на те, что позволяют эти операции.
    Решение с SFINAE:

        template <typename T = double, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
        class Tensor {

    Но я использую `concept`(С++20) и `is_arithmetic_v`.

    is_arithmetic_v - потенциальное место улучшения, потому что он разрешает только integral or floating_point,
    можно было бы проверять может ли тип проводить арифмитические операции, тогда Tensor смог бы хранить custom-ные типы,
    но это затруднит реализацию вычислений на GPU.
 */
template <Arithmetic T = double>
class Tensor {
protected:
    T _value;
    T * _coeffs;
    int * _shape; // Я думаю это излишне, и лучше вычислять shape recursively in runtime
    int _size; // better be unsigned int
    int _rank; // not mathematically correct name
public:
    /*
    Я хочу чтобы объект Tensor создавался так:

        Tensor tensor(depth, rows, cols);

    Для этого в конструкторе можно принимать Tensor(initializer_list),
    но тогда придется оборачивать размерность в {}:

        Tensor tensor({ depth, rows, cols });

    Еще хуже, что без explicit создавать объект можно будет так:

        Tensor tensor = { depth, rows, cols };

    И проблема с таким инстанциированием в том, что это интуитивно выглядит похоже будто итоговый tensor будет массивом (rank=1, size=3).

    Оказывается создать variadic arguments(int ... dims), как я делал в Java, в C++ нельзя,
    но что-то похожее можно реализовать с variadic template-ами.
    Возникает конечно необходимость в constraint-е принимать только целые значения размерностей, то есть int,
    для этого можно использовать Pack expansion со SFINAE: `std::conjunction_v<std::is_same<TArgs, int>...>`
    `conjunction` performing a logical AND, references:
    - https://en.cppreference.com/w/cpp/types/conjunction
    - https://en.cppreference.com/w/cpp/types/conditional

    SFINAE проверка:

        template <typename ... TArgs, typename = std::enable_if_t<(std::conjunction_v<std::is_same<TArgs, int>...>)>> // SFINAE
        Tensor(const TArgs ... dims) {

    Можно использовать `requires` clause из C++20:

        template <typename ... TArgs>
        requires std::conjunction_v<std::is_same<TArgs, int>...> // C++20
        Tensor(const TArgs ... dims) {

    Самым элегантным способом является abbreviated function templates, которые я и использую, references:
    - https://en.cppreference.com/w/cpp/language/variadic_arguments
    - https://federico-busato.github.io/Modern-CPP-Programming/11.Templates_II.pdf
    */
    explicit Tensor(std::integral auto ... dims) { // C++20, abbreviated function templates with concept
        int arr[] = {dims...}; // arguments expansion
        if(sizeof(arr) == 0) {
            throw "null tensor, scalar is at least of size 1";
        }
        _size = sizeof(arr) / sizeof(arr[0]);
        _shape = new int[]{dims...}; // arguments expansion
        _coeffs = new T[_size];
    }

    Tensor() : Tensor(1) {}

    // Tensor(std::initializer_list<T> list){}

    // --- Rule of 5 ---
    // Tensor лучше никогда не копировать и лучше применить rule of 5. И удалить copy constructor и copy assignment. ?
    ~Tensor(){
        delete[] _shape;
        delete[] _coeffs;
    }
    // Copy Constructor
    // Copy Assignment Operator
    // Move Constructor
    // Move Assignment Operator
    // ------
};

/*
    Specialization of std::is_arithmetic for custom class A.
    Может быть интересно, если хотите использовать custom-ный тип в Tensor.
    Если мы включим, то is_arithmetic_v<A> = true, и он будет проходить проверку SFINAE.
    I'd not recommend to do it, actually.

        class A {};

        template <typename T>
        class std::is_arithmetic<A> : public std::true_type {};

 */

int main(){
    std::cout << "Tensor.cpp execution started!" << std::endl;
    // int arr[0];
    // std::cout << sizeof(arr) << std::endl;

    Tensor tensor(1, 2, 3);
    // Tensor vector = {1, 2, 3};
    // Tensor matrix = {
    //     {1, 2, 3},
    //     {4, 5, 6},
    //     {7, 8, 9}
    // };
}

