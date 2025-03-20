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

#define DEBUG_LOG_TENSOR true

#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <concepts>

#include "utils/log.tpp"
#include "utils/Arithmetic.tpp"

namespace tensor {
/*
    Multidimensional Array with Mappings
 */
template <Arithmetic T = double>
class Tensor {
protected:
    T _value;
    int _rank; // not mathematically correct name, его тоже можно вычислить рекурсивно, оставляю для debug-а
    int _size; // better be unsigned int
    Tensor<T> * _coeffs;

public:
    explicit Tensor() : _value(0), _rank(0), _size(-1), _coeffs(nullptr) {}

    explicit Tensor(std::integral auto ... args){
        log("Basic Constructor");

        int dims[] = {args...}; // arguments expansion

        if(dims[0] <= 0){
            throw std::runtime_error("Error: Null tensor, dims[0] is invalid!");
        }

        int rank = sizeof(dims) == 0 ? 0 : sizeof(dims) / sizeof(dims[0]);

        __init(rank, dims, 0);
    }

    explicit Tensor(int rank, int dims[], int cursor) {
        log("Too Complex Constructor");

        if(rank < 0){
            throw std::runtime_error("Error: Null tensor, rank is invalid!");
        }
        __init(rank, dims, cursor);
    }

private:
    inline void __init(int rank, int dims[], int cursor){
        log("Init Constructor");

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
};

} // namespace tensor

/*
    Implementation of template class is in .tpp file
 */

#endif
