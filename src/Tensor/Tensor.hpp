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

#pragma once
#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <concepts>
#include <initializer_list>

namespace {
    template <typename T>
    concept Arithmetic = std::is_arithmetic_v<T>; // C++20
}

/*
    Я делаю Tensor homogeneous, то есть он хранит данные только одного типа.

    Можно реализовать Double-Tensor,
    но здесь реализовано решение получше сделать template Tensor,
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

private:
    explicit Tensor(int rank, int dims[], int cursor = 0);

    inline void __init(int rank, int dims[], int cursor);

protected:
    T _value;
    int _rank; // not mathematically correct name, его тоже можно вычислить рекурсивно, оставляю для debug-а
    int _size; // better be unsigned int
    Tensor * _coeffs;
    // int * _shape; // Я думаю это излишне и лучше вычислять shape recursively in runtime

public:
    using type = T;

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
    `conjunction` performing a logical AND,
    references:
        - https://en.cppreference.com/w/cpp/types/conjunction
        - https://en.cppreference.com/w/cpp/types/conditional

    SFINAE проверка:

        template <typename ... TArgs, typename = std::enable_if_t<(std::conjunction_v<std::is_same<TArgs, int>...>)>> // SFINAE
        Tensor(const TArgs ... dims) {

    Можно использовать `requires` clause из C++20:

        template <typename ... TArgs>
        requires std::conjunction_v<std::is_same<TArgs, int>...> // C++20
        Tensor(const TArgs ... dims) {

    Самым элегантным способом является "abbreviated function templates", которые я и использую,
    references:
        - https://en.cppreference.com/w/cpp/language/variadic_arguments
        - https://federico-busato.github.io/Modern-CPP-Programming/11.Templates_II.pdf
    */
    explicit Tensor(std::integral auto ... args); // C++20, abbreviated function templates with concept

    explicit Tensor() : _value(0), _rank(0), _size(-1), _coeffs(nullptr) {}

    // --- initializer_list
    Tensor(const std::initializer_list<T> list);
    Tensor(const std::initializer_list<std::initializer_list<T>> list);
    Tensor(const std::initializer_list<Tensor<T>> list);
    // ------

    // --- Rule of 5 ---

    // Tensor лучше никогда не копировать и лучше применить rule of 5. И удалить copy constructor и copy assignment. ?

    virtual ~Tensor(); // don't know yet will there be inheritance from Tensor, probably it's okay to make destructor virtual

    // Copy Constructor
    Tensor(const Tensor<T> & other) = delete;

    // Copy Assignment Operator
    Tensor<T> & operator = (const Tensor<T> & other) = delete;

    // Move Constructor
    Tensor(Tensor<T> && other);

    // Move Assignment Operator
    Tensor<T> & operator = (Tensor<T> && other); // tensor[index] = Tensor();

    // Copy Assignment Operator (Scalar)
    Tensor<T> & operator = (const T & scalar); // tensor[index] = number;

    // ------

    inline T value() const { return _value; }

    inline bool isScalar(){ return _rank == 0 && _size == -1 && _coeffs == nullptr; }

    inline int size(){ return _size; }

    /*
    Java style accessing through methods:

        inline Tensor& get(int i, int ... indices){
            return _coeffs[i].get(indices...);
        }
        inline T get(int i, int ... indices){
            return _coeffs[i].get(indices...).value()
        }
    */

    // --- Operator Overloadings ---

    // Index Operator []
    inline Tensor& operator [] (const int index) const { return _coeffs[index]; }

    // Typecast overloading
    operator T () const { return _value; }

    // +=
    // +

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


/*
    Implementation of template class is in .tpp file
 */
#include "Tensor.tpp"


#endif
