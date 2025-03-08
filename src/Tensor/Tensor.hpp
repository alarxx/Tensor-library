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

// Макрос для создания вложенных std::initializer_list
#define INITIALIZER_LIST_1(T) std::initializer_list<T>
#define INITIALIZER_LIST_2(T) std::initializer_list<INITIALIZER_LIST_1(T)>
#define INITIALIZER_LIST_3(T) std::initializer_list<INITIALIZER_LIST_2(T)>
#define INITIALIZER_LIST_4(T) std::initializer_list<INITIALIZER_LIST_3(T)>

#include <iostream>
#include <cassert>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <concepts>
#include <initializer_list>
#include <vector>

#include <iterator>
#include <cstddef> // ptrdiff_t
#include "iterator/iterator.hpp"
#include "iterator/constant_iterator.hpp"

#include "utils/is_vector.tpp"
#include "utils/nested_vector_info.tpp"
#include "utils/Arithmetic.tpp"

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
        Tensor(const TArgs ... dims);

        template <Arithmetic T> // Definition
        template <typename ... TArgs, std::enable_if_t<(std::is_same_v<TArgs, int> && ... && true), int> = 0>
        Tensor<T>::Tensor(TArgs ... args) {}

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
    explicit Tensor(auto ... args){
        throw std::runtime_error("Incorrect type: only ints in explicit Tensor(int ... dims)");
    }

    explicit Tensor(int rank, int dims[], int cursor = 0);

    explicit Tensor() : _value(0), _rank(0), _size(-1), _coeffs(nullptr) {}

    /*
        Tensor from any dimensional raw array

        Нужно возвращать tensor of array type, но этот тип нужно вытаскивать рекурсивно, i.e. int[][] -> int:
            std::cout << typeid(decltype(arr)).name() << std::endl; // int[][]
            std::cout << typeid(std::decay_t<decltype(arr)>).name() << std::endl; // int[]
            std::cout << typeid(std::remove_all_extents_t<decltype(arr)>).name() << std::endl; // recursively returns primitive type

        Initial enter via from_array function, and the then it goes to recursive _from_array, which should be private, actually, but it's a friend function, so I couldn't make it private.
        from_array uses _from_array, so _from_array must be declared and implemented first.
     */
    template <typename U>
    requires Arithmetic<U>
    /*private*/ friend Tensor<U> _from_array(
        const std::vector<int>& dims,
        void * arr,
        const long unsigned int cursor
    );
    template <typename U, int SIZE>
    requires Arithmetic<std::remove_all_extents_t<U>> // По идее не обязательно здесь делать эту проверку, дальше Tensor<?> проверит
    friend Tensor<std::remove_all_extents_t<U>> from_array(
        const std::vector<int>& dims,
        U (&arr)[SIZE]
    );

    // Tensor from vector
    template <typename U>
    requires Arithmetic<nested_vector_info_t<U>>
    friend Tensor<nested_vector_info_t<U>>
    from_stl_vector(
        std::vector<U>& vector
    );

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

    /*
        Как создать tensor из множества тензоров.

        SFINAE для проверки соответствия типов тензоров.

        std::cout << "&&: " << std::is_same_v<std::remove_reference_t<Tensor<int>&&>, Tensor<int>> << std::endl; // true

        Tensor tensor = {1, 2, 3};
        Tensor<int> * tptr = &tensor;
        Tensor<int> & tref = *tptr;
        std::cout << std::is_same_v<decltype(tref), Tensor<typename decltype(tensor)::type>> << std::endl; // without remove_reference_t
        std::cout << std::is_same_v<std::remove_reference_t<decltype(tref)>, Tensor<typename decltype(tensor)::type>> << std::endl;
    */
    // Copy concat constructor
    template <typename ... TArgs>
    requires (std::is_same_v<std::remove_reference_t<TArgs>, Tensor<T>> && ... && true) // requires all args to be tensors of the same type
    /*explicit*/ Tensor(const Tensor<T>& first, const TArgs& ... args);
    // Move concat constructor
    template <typename ... TArgs>
    requires (std::is_same_v<std::remove_reference_t<TArgs>, Tensor<T>> && ... && true) // requires all args to be tensors of the same type
    /*explicit*/ Tensor(Tensor<T>&& first, TArgs&& ... args);

    // --- initializer_list
    /*
        initializer_list используется для nice syntax-а создания tensor-ов:

            Tensor matrix = {
                {1, 2, 3},
                {1, 2, 3},
                {1, 2, 3}
            };

        Кажется, через templates можно создать бесконечно рекурсивный initializer_list принимающий любые rank-и: { { { {{1, 2, 3},}, }, }, ... } ?
        Нет, нельзя, кажется, никак нельзя, компилятор не может deduct U в initializer_list<U> при передаче {{1, 2}, {3, 4}}.
     */
    Tensor(const INITIALIZER_LIST_1(T) list);
    Tensor(const INITIALIZER_LIST_2(T) list);
    Tensor(const INITIALIZER_LIST_3(T) list);
    Tensor(const INITIALIZER_LIST_4(T) list);
    // Tensor(const std::initializer_list<Tensor<T>> list); // concat {tensors}
    // ------

    // --- concat ---

    // Copy concat
    template <Arithmetic U, typename ... TArgs>
    // requires (std::is_same_v<TArgs, Tensor<U>> && ... && true) // For some reason it allows Tensor& and Tensor&& in comparison to Tensor.
    requires (std::is_same_v<std::remove_reference_t<TArgs>, Tensor<U>> && ... && true)
    friend Tensor<U> concat(const Tensor<U>& first, const TArgs& ... tensors);

    // Move concat
    template <Arithmetic U, typename ... TArgs>
    requires (std::is_same_v<std::remove_reference_t<TArgs>, Tensor<U>> && ... && true)
    friend Tensor<U> concat(const Tensor<U>&& first, const TArgs&& ... tensors);

    template <Arithmetic U>
    friend Tensor<U> concat(const int size, const Tensor<U> tensors[]);

    // ------

    // --- Rule of 5 ---

    // Tensor лучше никогда не копировать и лучше применить rule of 5. И удалить copy constructor и copy assignment. ?

    /*virtual*/ ~Tensor();
    // don't know yet will there be inheritance from Tensor, probably it's okay to make destructor virtual
    // virtual добавляет 1 указатель на vtable (8 byte), поэтому без virtual.

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

    inline T& value() {
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
    inline Tensor copy() const { return Tensor(*this); }
    inline Tensor move() { return Tensor(std::move(*this)); }

private:
    inline void _shape(Tensor<T>& t, std::vector<int>& vec){
        if(t._rank == 0)
            return;
        vec.push_back(t._size);
        _shape(t[0], vec);
    }
public:
    inline std::vector<int> shape(){
        std::vector<int> vec;
        _shape(*this, vec);
        return vec;
    }

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
            // buffer overflow, нужно ли делать эту проверку
            // Я думаю, лучшим решением будет user-у просто проверять размеры, а не проверять тут миллион раз
            if(index >= _size){
                throw std::runtime_error("Index out of bounds!");
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

    // --- iterator ---

    using iterator = ::iterator<Tensor<T>>; // iterator variable shadowing, so we use :: - global namespace.
    using constant_iterator = ::constant_iterator<Tensor<T>>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using constant_reverse_iterator = std::reverse_iterator<constant_iterator>;

    // _coeffs[0] = *(_coeffs + 0)
    // &(*(_coeffs + 0)) = _coeffs
    iterator begin(){ return iterator(_coeffs); }
    iterator end(){ return iterator(&_coeffs[_size]); }

    constant_iterator cbegin() const { return constant_iterator(&_coeffs[0]); }
    constant_iterator cend() const { return constant_iterator(&_coeffs[_size]); }

    reverse_iterator rbegin(){ return reverse_iterator(end()); }
    reverse_iterator rend(){ return reverse_iterator(begin()); }

    constant_reverse_iterator crbegin() const {
        return constant_reverse_iterator(cend()); // &_coeffs[N - 1]
    }
    constant_reverse_iterator crend() const {
        return constant_reverse_iterator(cbegin()); // &_coeffs[-1]
    }

    // ------

};

/*
    Implementation of template class is in .tpp file
 */
#include "impl/log.tpp"
#include "impl/constructor.tpp"
#include "impl/concat_constructor.tpp"
#include "impl/concat.tpp"
#include "impl/initializer_list.tpp"
#include "impl/io.tpp"
#include "impl/arithmetic_operator.tpp"
#include "impl/from_array.tpp"
#include "impl/from_stl_vector.tpp"

#endif
