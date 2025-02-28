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

#define DEBUG_TENSOR true
#define DEBUG_LOG_TENSOR true

#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <concepts>
#include <initializer_list>

// namespace {
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>; // C++20
// }

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

    std::string __toString() const;

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

    И проблема с таким инстанциированием в том, что это интуитивно выглядит похоже будто итоговый tensor будет вектором (rank=1, size=3).

    Сейчас initializer_list используется для nice syntax-а создания tensor-ов:

        Tensor matrix = {
            {1, 2, 3},
            {1, 2, 3},
            {1, 2, 3}
        };

    ---------------------------------------------------------------------------

    Оказывается, создать variadic arguments(int ... dims), как я делал в Java, в C++ нельзя,
    но что-то похожее можно реализовать с variadic template-ами.
    Возникает конечно необходимость в constraint-е принимать только целые значения размерностей, то есть int,
    для этого можно использовать Pack expansion со SFINAE: `std::conjunction_v<std::is_same<TArgs, int>...>`
    `conjunction` performing a logical AND,
    references:
        - https://en.cppreference.com/w/cpp/types/conjunction
        - https://en.cppreference.com/w/cpp/types/conditional

    SFINAE проверка:

        template <typename ... TArgs, std::enable_if_t<(std::is_same_v<TArgs, int> && ... && true), int> = 0> // SFINAE
        Tensor(const TArgs ... dims) {

    Wrong way to achieve SFINAE:

        // may cause redefinition error
        template <typename ... TArgs, typename = std::enable_if_t<(std::conjunction_v<std::is_same<TArgs, int>...>)>>

    reference:
        - https://github.com/federico-busato/Modern-CPP-Programming/issues/183

    ---------------------------------------------------------------------------

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
    Tensor(const std::initializer_list<Tensor<T>> list); // appending {tensors}
    // ------

    /*
        Нужен был простой способ создать скаляр.

        Конструктор explicit Tensor(int ... args), поэтому мы не можем вызывать в форме:

            Tensor tensor = 3; // Error

        Это могло значить:

            Tensor tensor(3); // creates vector of size 3

        Но это создает вектор (1D), размером в 3 элемента.
    */
    template <Arithmetic U>
    friend Tensor<U> scalar(U value);

    // --- Rule of 5 ---

    // Tensor лучше никогда не копировать и лучше применить rule of 5. И удалить copy constructor и copy assignment. ?

    virtual ~Tensor(); // don't know yet will there be inheritance from Tensor, probably it's okay to make destructor virtual

    // Copy Constructor
    Tensor(const Tensor<T> & other);

    // Copy Assignment Operator
    Tensor<T> & operator = (const Tensor<T> & other);

    // Move Constructor
    Tensor(Tensor<T> && other);

    // Move Assignment Operator
    Tensor<T> & operator = (Tensor<T> && other); // tensor[index] = Tensor();

    // Copy Assignment Operator (Scalar)
    Tensor<T> & operator = (const T & scalar); // tensor[index] = number;

    // ------

    inline T value() const {
        #if DEBUG_TENSOR
            if(!isScalar()){
                throw std::runtime_error("Error: Geting a value of a non-scalar tensor!");
            }
        #endif
        return _value;
    }

    inline bool isScalar() const { return _rank == 0 && _size == -1 && _coeffs == nullptr; }
    inline bool isVector() const { return _rank == 1 && _size != -1 && _coeffs != nullptr; }
    inline bool isMatrix() const { return _rank == 2 && _size != -1 && _coeffs != nullptr; }
    // Может ли вектор иметь size = 0

    inline int size() const { return _size; }
    inline int rank() const { return _rank; }

    std::string toString() const;

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
    inline Tensor& operator [] (const int index) {
        #if DEBUG_TENSOR
            if(isScalar()){
                throw std::runtime_error("Can't access scalar tensor by index");
            }
        #endif
        return _coeffs[index];
    }

    // Typecast overloading
    // operator T () const {
    //     if(!isScalar()){
    //         throw std::runtime_error("Error: Can't typecast non-scalar tensor!");
    //     }
    //     return _value;
    // }

    // Stream insertion operation
    template <Arithmetic U>
    friend std::ostream& operator<<(std::ostream& os, const Tensor<U>& tensor);

    // +=
    // +

    // Unary Operator
    // Elementwise: v1 *= v2;
    Tensor<T>&  operator *= (const Tensor<T>& other);

    // Binary Operator
    // friend - не является членом класса, но имеет доступ к private
    // Note: no "self" vector argument, therefore we use "friend" keyword
    template <typename U>
    friend Tensor<U> operator * (const Tensor<U> & /*const*/ v1, const Tensor<U> & /*const*/ v2);

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
