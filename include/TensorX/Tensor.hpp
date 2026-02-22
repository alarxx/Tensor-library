/*
    SPDX-License-Identifier: MPL-2.0
    --------------------------------
    This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
    If a copy of the MPL was not distributed with this file,
    You can obtain one at https://mozilla.org/MPL/2.0/.

    This file is part of the Tensor-library:
    https://github.com/alarxx/Tensor-library

    Provided “as is”, without warranty of any kind.

    Copyright © 2022-2026 Alar Akilbekov. All rights reserved.
*/

#pragma once
#ifndef TENSOR_HPP
#define TENSOR_HPP

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

namespace tensor {

template <typename T>
concept Arithmetic = requires(T a, T b){
    // requires std::is_arithmetic_v<T>;

    // specific check for what we need
    {a += b};
    {a -= b};
    {a *= b};
    {a /= b};

    { a + b } -> std::same_as<T>;
    { a - b } -> std::same_as<T>;
    { a * b } -> std::same_as<T>;
    { a / b } -> std::same_as<T>;
};


template <Arithmetic T = float>
class Tensor {
private:
    int rank; // not mathematically correct name, его тоже можно вычислить рекурсивно, оставляю для debug-а
    int * dims; // shape
    T * coeffs;
public:
    using type = T;

    /*
    0D: (0, {}) - scalar
    1D: (1, {4}) - vector
    2D: (2, {4, 4}) - matrix
    3D: (3, {4, 4, 4})
    */
    explicit Tensor() : rank(0), dims(nullptr), coeffs(new T[1]) {
        // scalar
    }

    // --- scalar(value) ---
    // Tensor sc = scalar(42.0);
    template <Arithmetic U>
    friend Tensor<U> scalar(U value);
    // ------

    explicit Tensor(int rank, int dims[]);

    explicit Tensor(std::integral auto ... args); // C++20, abbreviated function templates with concept
    explicit Tensor(auto ... args){
        throw std::runtime_error("Incorrect type: only ints in explicit Tensor(int ... dims)");
    }

    // --- initializer_list ---
    Tensor(const INITIALIZER_LIST_1(T) list);
    Tensor(const INITIALIZER_LIST_2(T) list);
    Tensor(const INITIALIZER_LIST_3(T) list);
    Tensor(const INITIALIZER_LIST_4(T) list);
    // ------


    // --- Rule of 5 ---
    // Destructor
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
    // ------

    // --- Index Operator [] ---
    // The idea is to call tensor[d][r][c], which would be nice.
    // Here is the problem with proxy "TensorView" which should not make copy, or should idk.
    // inline Tensor& operator [] (const int index) {
    //     return coeffs[index];
    // }
    inline T& get(std::integral auto ... args);

    // --- toString ---
    std::string toString() const;

    // Stream insertion operation <<
    template <Arithmetic U>
    friend std::ostream& operator<<(std::ostream& os, const Tensor<U>& tensor);


    // --- ---
    int getRank() const { return rank; }

    int * getDims() const { return dims; }

    T * getCoeffs() { return coeffs; }

    int getLength() const {
        int s = 1;
        for (int d = 0; d < rank; ++d){
            s *= dims[d];
        }
        return s;
    }
};


// --- scalar(value) ---
// Tensor sc = scalar(42.0);
template <Arithmetic U>
/*friend*/ Tensor<U> scalar(U value){
    Tensor<U> tensor;
    *tensor.coeffs = value; // tensor[0]
    return tensor; // RVO
}
// ------


template <Arithmetic U>
std::string Tensor<U>::toString() const {
    // print undefined tensor
    if(rank < 0 && dims == nullptr && coeffs == nullptr){
        return "tensor(null)";
    }
    std::string res = "";
    res += "tensor(" + std::to_string(rank) + "D)<" + std::string(typeid(coeffs[0]).name()) + ">:\n";
    int size = 1; for(int i = 0; i < rank; i++) size *= dims[i];
    for(int i = 0; i < size; i++){
        if(rank >= 1 && i % dims[rank - 1] == 0 && i != 0){
            res += "\n";
        }
        if(rank >= 2 && i % (dims[rank - 1] * dims[rank - 2]) == 0 && i != 0){
            res += "\n";
        }
        res += " " + std::to_string(coeffs[i]);
    }
    return res;
}

// Stream insertion operation <<
template <Arithmetic U>
/*friend*/ std::ostream& operator << (std::ostream& os, const Tensor<U>& tensor){
    os << tensor.toString();
    return os;
}

template <Arithmetic U>
Tensor<U>::Tensor(int rank, int dims[]){ // int dims[] is int * dims
    if(rank < 0){
        throw std::runtime_error("Error: Null tensor, rank is invalid!");
    }
    if(dims == nullptr){
        throw std::runtime_error("Error: Null tensor, dims is invalid!");
    }
    this->rank = rank;
    this->dims = new int[rank];
    for(int i = 0; i < rank; i++){
        this->dims[i] = dims[i];
    }

    int size = 1; for(int i = 0; i < rank; i++) size *= dims[i];
    coeffs = new U[size];
}

// Constructor : Tensor tensor(depth, rows, cols)
template <Arithmetic U>
Tensor<U>::Tensor(std::integral auto ... args) { // C++20, abbreviated function templates with concept
    int _dims[] = {args...}; // arguments expansion
    // rank = sizeof(dims) == 0 ? 0 : sizeof(dims) / sizeof(dims[0]); // number of dim entries
    rank = sizeof...(args); // number of dim entries
    dims = new int[rank];
    for(int i = 0; i < rank; i++){
        dims[i] = _dims[i];
    }
    int size = 1; for(int i = 0; i < rank; i++) size *= dims[i];
    coeffs = new U[size];
}


// --- initializer_list ---

// Tensor vector = {1, 2, 3};
template <Arithmetic T>
Tensor<T>::Tensor(const INITIALIZER_LIST_1(T) list){
    // 1D: {0, 1, 2}
    rank = 1;
    int size = list.size();
    dims = new int[rank]{size};
    coeffs = new T[size];
    int i = 0;
    for(auto e: list){
        // we can't access initializer_list by index, only by range-based for loop or by iterator
        coeffs[i++] = e;
    }
}

// Tensor matrix = {{1, 2}, {3, 4}};
template <Arithmetic T>
Tensor<T>::Tensor(const INITIALIZER_LIST_2(T) list){
    // 2D:
    // {{0, 1, 2},
    //  {3, 4, 5},
    //  {6, 7, 8}}
    rank = 2;
    int rows = list.size(),
        cols = list.begin()->size();
    dims = new int[rank]{rows, cols};
    int size = rows * cols;
    coeffs = new T[size];
    // we can't access initializer_list by index, only by range-based for loop or by iterator
    int i = 0;
    for(auto & r: list){
        for(auto c: r){
            // std::cout << c << std::endl;
            coeffs[i++] = c;
        }
    }
}

template <Arithmetic T>
Tensor<T>::Tensor(const INITIALIZER_LIST_3(T) list){
    rank = 3;
    int depth = list.size(),
        rows = list.begin()->size(),
        cols = list.begin()->begin()->size();
    dims = new int[rank]{depth, rows, cols};
    int size = depth * rows * cols;
    coeffs = new T[size];
    // we can't access initializer_list by index, only by range-based for loop or by iterator
    int i = 0;
    for(auto & d: list){
        for(auto & r: d){
            for(auto c: r){
                // std::cout << c << std::endl;
                coeffs[i++] = c;
            }
        }
    }
}

template <Arithmetic T>
Tensor<T>::Tensor(const INITIALIZER_LIST_4(T) list){
    rank = 4;
    int batch = list.size(),
        depth = list.begin()->size(),
        rows = list.begin()->begin()->size(),
        cols = list.begin()->begin()->begin()->size();
    dims = new int[rank]{batch, depth, rows, cols};
    int size = batch * depth * rows * cols;
    coeffs = new T[size];
    // we can't access initializer_list by index, only by range-based for loop or by iterator
    int i = 0;
    for(auto & b : list){
        for(auto & d: b){
            for(auto & r: d){
                for(auto c: r){
                    // std::cout << c << std::endl;
                    coeffs[i++] = c;
                }
            }
        }
    }
}

// ------


// --- Rule of 5 ---

template <Arithmetic T>
Tensor<T>::~Tensor(){
    std::cout << "Destructor" << std::endl;
    rank = -1;
    if(coeffs != nullptr){
        delete[] coeffs;
        coeffs = nullptr;
    }
    if(dims != nullptr){
        delete[] dims;
        dims = nullptr;
    }
}


// Copy Constructor
// O(N)
template <Arithmetic T>
Tensor<T>::Tensor(const Tensor<T> & other) {
    std::cout << "Copy Constructor" << std::endl;

    /*
    No need in delete[] coeffs and dims, because it is a constructor
    */

    rank = other.rank;

    int size = 1;

    if(rank > 0 && other.dims != nullptr){
        dims = new int[rank];
        for(int i = 0; i < rank; i++){
            dims[i] = other.dims[i];
            size *= other.dims[i];
        }
    }

    coeffs = new T[size];

    for(int i = 0; i < size; i++){
        coeffs[i] = other.coeffs[i];
    }
}


// Copy Assignment Operator
// O(N)
template <Arithmetic T>
Tensor<T> & Tensor<T>::operator = (const Tensor<T> & other){
    std::cout << "Copy Assignment Operator" << std::endl;
    if(this != &other){
        /*
        // Copy assignment should not require rank and dims match,
        // users should check it themself if needed.

        if (rank != other.rank){
            throw std::runtime_error("Error: rank doesn't match for copy assignment");
        }
        if(dims[i] != other.dims[i]){
            throw std::runtime_error("Error: dims doesn't match for copy assignment");
        }
        */

        rank = -1;
        if(coeffs != nullptr){
            delete[] coeffs;
            coeffs = nullptr;
        }
        if(dims != nullptr){
            delete[] dims;
            dims = nullptr;
        }

        rank = other.rank;

        int size = 1;
        if(rank > 0 && other.dims != nullptr){
            dims = new int[rank];
            for(int i = 0; i < rank; i++){
                dims[i] = other.dims[i];
                size *= other.dims[i];
            }
        }

        coeffs = new T[size];
        for(int i = 0; i < size; i++){
            coeffs[i] = other.coeffs[i];
        }
    }
    return *this;
}


// Move Constructor
// O(1)
template <Arithmetic T>
Tensor<T>::Tensor(Tensor<T> && other){
    std::cout << "Move Constructor" << std::endl;
    // this = other
    rank = other.rank;
    dims = other.dims;
    coeffs = other.coeffs;
    // other = null, reset but not delete
    other.rank = -1;
    other.dims = nullptr;
    other.coeffs = nullptr;
}


// Move Assignment Operator
// O(1)
template <Arithmetic T>
Tensor<T> & Tensor<T>::operator = (Tensor<T> && other){
    std::cout << "Move Assignment Operator" << std::endl;
    if(this != &other){
        /*
        // Copy assignment should not require rank and dims match,
        // users should check it themself if needed.
        if (rank != other.rank){
            throw std::runtime_error("Error: rank doesn't match for move assignment");
        }
        for(int i = 0; i < rank; i++){
            if(dims[i] != other.dims[i]){
                throw std::runtime_error("Error: dims doesn't match for move assignment");
            }
        }
        */
        rank = -1;
        if(coeffs != nullptr){
            delete[] coeffs;
            coeffs = nullptr;
        }
        if(dims != nullptr){
            delete[] dims;
            dims = nullptr;
        }

        // this = other
        rank = other.rank;
        dims = other.dims;
        coeffs = other.coeffs;
        // other = null, reset but not delete
        other.rank = -1;
        other.dims = nullptr;
        other.coeffs = nullptr;

    }
    return *this;
}

// ------


template <Arithmetic U>
inline U& Tensor<U>::get(std::integral auto ... args){
    int _dims[] = {args...}; // arguments expansion (not compute free operation)
    if(rank != sizeof...(args)){ // number of dim entries
        throw std::runtime_error("Error: incorrect number of indexes");
    }
    int index = 0; // mapping
    int stride = 1;
    for(int d = rank - 1; d >= 0; --d){
        if(d != rank - 1){
            stride *= dims[d + 1];
        }
        index += _dims[d] * stride;
    }
    return coeffs[index];
}

} // namespace tensor

#endif
