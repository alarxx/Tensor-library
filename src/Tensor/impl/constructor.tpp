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

namespace tensor {

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
/*private*/ Tensor<T>::Tensor(int rank, int dims[], int cursor) {
    if(rank < 0){
        throw std::runtime_error("Error: Null tensor, rank is invalid!");
    }
    __init(rank, dims, cursor);
}


template <Arithmetic T>
/*private*/ inline void Tensor<T>::__init(int rank, int dims[], int cursor){
    // 0D: (0, {})
    // 1D: (1, {4})
    // 2D: (2, {4, 4})
    // 3D: (3, {4, 4, 4})
    log(cursor, ") rank-", rank, " Constructor of Tensor");

    _rank = rank;
    _size = dims[cursor];
    if(_rank < 0){
        throw std::runtime_error("Error: rank is invalid!");
    }
    else if(_size <= 0){
        throw std::runtime_error("Error: size is invalid!");
    }

    // --- recursively initialization of nested tensors ---
    _coeffs = new Tensor<T>[_size];
    // after this, tensors look like scalars, but it's okay we'll move rvalues into them

    if(rank - 1 == 0) { // then it is a Scalar
        log("Next is scalar tensor, _size:(", _size, "), so we return");
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


// --- scalar(value) ---
/*
    Tensor sc = scalar(42.0);
 */
template <Arithmetic U>
/*friend*/ Tensor<U> scalar(U value){
    Tensor<U> tensor;
    tensor = value;
    // RVO
    return tensor;
}
// ------


// --- Rule of 5 ---

template <Arithmetic T>
Tensor<T>::~Tensor(){
    log("~ rank-", _rank, " Destructor of Tensor", (isScalar() ? " (Scalar)" : ""));
    if(_coeffs != nullptr){
        log("\tdelete[] coeffs");
        // рекурсивно удаляются nested tensors
        delete[] _coeffs;
        _coeffs = nullptr;
    }
}


// Copy Constructor
// O(N)
template <Arithmetic T>
Tensor<T>::Tensor(const Tensor<T> & other) : Tensor() {
    log("Copy Constructor", (other.isScalar() ? " (Scalar)" : ""), " (rank=", other._rank, ", size=", other._size, ")");

    _rank = other._rank;
    _size = other._size;
    // Мы можем назначать и scalar tensor, не только используя .value() method to assign scalar
    _value = other._value;

    if(_rank > 0){
        _coeffs = new Tensor<T>[_size];

        for(int i = 0; i < _size; i++){
            // Чтобы он не выглядел как scalar, мы назначаем _rank и _size
            _coeffs[i]._rank = other._coeffs[i]._rank;
            _coeffs[i]._size = other._coeffs[i]._size;
            _coeffs[i] = other._coeffs[i]; // Copy Assignment =
        }
    }
}


// Copy Assignment Operator
// O(N)
template <Arithmetic T>
Tensor<T> & Tensor<T>::operator = (const Tensor<T> & other){
    log("Copy Assignment Operator", (isScalar() ? " (Scalar)" + std::to_string(other._value) : ""), " (rank=", other._rank, "->", _rank, ", size=", other._size, "->", _size, ")");

    if(this != &other){

        if (_rank != other._rank){
            throw std::runtime_error("Error: rank doesn't match for copy assignment");
        }
        if(_size != other._size){
            throw std::runtime_error("Error: size doesn't match for copy assignment");
        }

        if(_coeffs != nullptr){
            log("\tdelete[] coeffs");
            delete[] _coeffs;
            _coeffs = nullptr;
        }

        // _rank и _size уже одинаковые
        // _rank = other._rank;
        // _size = other._size;
        // Мы можем назначать и scalar tensor, не только используя .value() method to assign scalar
        _value = other._value;

        if(_rank > 0){ // if _rank == 0 tensor is scalar
            _coeffs = new Tensor<T>[_size];

            for(int i = 0; i < _size; i++){
                // Чтобы он не выглядел как scalar, мы назначаем _rank и _size
                _coeffs[i]._rank = other._coeffs[i]._rank;
                _coeffs[i]._size = other._coeffs[i]._size;
                _coeffs[i] = other._coeffs[i]; // Copy Assignment =
            }
        }

        // no need to delete other
    }

    return *this;
}


// Move Constructor
// O(1)
template <Arithmetic T>
Tensor<T>::Tensor(Tensor<T> && other){
    log("Move Constructor", (isScalar() ? " (Scalar)" : ""), " (rank=", other._rank, ", size=", other._size, ")");
    // this = other
    _value = other._value;
    _rank = other._rank;
    _size = other._size;
    _coeffs = other._coeffs;
    // other = null (looks like scalar)
    other._value = 0;
    other._rank = 0;
    other._size = -1;
    other._coeffs = nullptr;
}


// Move Assignment Operator
// O(1)
template <Arithmetic T>
Tensor<T> & Tensor<T>::operator = (Tensor<T> && other){
    log("Move Assignment Operator", (isScalar() ? " (Scalar)" : ""), " (rank=", other._rank, "->", _rank, ", size=", other._size, "->", _size, ")");

    if(this != &other){

        if (_rank != other._rank){
            throw std::runtime_error("Error: rank doesn't match for move assignment");
        }
        if(_size != other._size){
            throw std::runtime_error("Error: size doesn't match for move assignment");
        }

        if(_coeffs != nullptr){
            log("\tdelete[] coeffs");
            delete[] _coeffs;
            _coeffs = nullptr;
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
// O(1)
template <Arithmetic T>
Tensor<T> & Tensor<T>::operator = (const T & scalar){
    if(!isScalar()){
        throw std::runtime_error("Error: Can't assign scalar to a non-scalar tensor!");
    }
    log("rank-", _rank, " Copy Scalar Assignment Operator: ", scalar, typeid(T).name());
    _value = scalar;
    return *this;
}

// ------

} // namespace tensor
