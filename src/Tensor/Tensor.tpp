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

#include "Tensor.hpp"


// --- Logging ---
#define DEBUG_LOG_TENSOR true

namespace {
    template <typename T>
    concept is_stream_supported = requires (T t){
        // SFINAE constrains
        std::declval<std::ostream&>() << t;
    };

    template <typename T>
    requires is_stream_supported<T>
    void log(T t){
        #if DEBUG_LOG_TENSOR
            std::cout << t << std::endl;
        #endif
    }

    template <typename T, typename ... TArgs>
    requires is_stream_supported<T>
    void log(T t, TArgs ... args){
        #if DEBUG_LOG_TENSOR
            std::cout << t;
            log(args...);
        #endif
    }
}
// ------


// Constructor : Tensor tensor(depth, rows, cols)
template <Arithmetic T>
Tensor<T>::Tensor(std::integral auto ... args) { // C++20, abbreviated function templates with concept
    int dims[] = {args...}; // arguments expansion

    if(dims[0] <= 0){
        throw std::runtime_error("Error: Null tensor, dims[0] is invalid!");
    }

    int rank = sizeof(dims) == 0 ? 0 : sizeof(dims) / sizeof(dims[0]);

    __init(rank, dims, 0);
}


template <Arithmetic T>
Tensor<T>::Tensor(int rank, int dims[], int cursor) {
    if(rank < 0){
        throw std::runtime_error("Error: Null tensor, rank is invalid!");
    }
    __init(rank, dims, cursor);
}


template <Arithmetic T>
inline void Tensor<T>::__init(int rank, int dims[], int cursor){
    // 0D: (0, {})
    // 1D: (1, {4})
    // 2D: (2, {4, 4})
    // 3D: (3, {4, 4, 4})
    log(cursor, ") rank-", rank, " Constructor of Tensor");

    _rank = rank;
    _size = dims[cursor];

    // --- recursively initialization of nested tensors ---
    _coeffs = new Tensor<T>[_size];
    // after this, tensors look like scalars, but it's okay we'll move rvalues into them

    if(rank - 1 == 0) { // then it is a Scalar
        log("Next is scalar tensor(", _size, "), so we return");
        return;
    }

    for(int i = 0; i < _size; i++){
        // Чтобы он не выглядел как scalar, мы назначаем _rank и _size
        _coeffs[i]._rank = rank - 1;
        _coeffs[i]._size = dims[cursor + 1];
        _coeffs[i] = Tensor(rank - 1, dims, cursor + 1); // Move Assignment =
    }
    // ------
}


// template <Arithmetic T>
// Tensor<T>::Tensor(std::initializer_list<T> list){}


// --- Rule of 5 ---

template <Arithmetic T>
Tensor<T>::~Tensor(){
    log("~ rank-", _rank, " Destructor of Tensor", (isScalar() ? " (Scalar)" : ""));
    if(_coeffs != nullptr){
        log("\tdelete[] coeffs");
        // рекурсивно удаляются nested tensors
        delete[] _coeffs;
    }
}


// Copy Constructor
// Copy Assignment Operator
// Move Constructor
// Move Assignment Operator
template <Arithmetic T>
Tensor<T> & Tensor<T>::operator = (Tensor<T> && other){
    log("Move Assignment Operator", (isScalar() ? " (Scalar)" : ""), " (rank=", other._rank, ", size=", _size, ")");

    if(this != &other){

        if (_rank != other._rank){
            log("Move from rank=", other._rank, " to rank=", _rank);
            throw std::runtime_error("Error: rank doesn't match for move assignment");
        }
        if(_size != other._size){
            log("Move from size=", other._size, " to size=", _size);
            throw std::runtime_error("Error: size doesn't match for move assignment");
        }

        if(_coeffs != nullptr){
            log("\tdelete[] coeffs");

            delete[] _coeffs;
        }

        // _rank и _size уже одинаковые
        // _rank = other._rank;
        // _size = other._size;
        // Мы можем назначать и scalar tensor, не только используя .value() method to assign scalar
        _value = other._value;
        _coeffs = other._coeffs;

        // reset but not delete, будет выглядеть как scalar
        other._value = 0;
        other._rank = 0;
        other._size = -1;
        other._coeffs = nullptr;

    }

    return *this;
}


// Copy Scalar Assignment Operator
template <Arithmetic T>
Tensor<T> & Tensor<T>::operator = (const T & scalar){
    if(!isScalar()){
        throw std::runtime_error("Error: Can't assign scalar to a non-scalar tensor!");
    }
    log("rank-", _rank, " Move Scalar Assignment Operator");
    _value = scalar;
    return *this;
}


// ------

