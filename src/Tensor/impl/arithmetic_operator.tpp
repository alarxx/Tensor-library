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

// Unary Operator
// Elementwise: v1 *= v2;
template <Arithmetic T>
Tensor<T>& Tensor<T>::operator *= (const Tensor<T>& other){
    #if DEBUG_TENSOR
        // Different sizes may cause overflow, нужно ли делать эту проверку
        // Я думаю, лучшим решением будет user-у просто перед умножением проверять размеры, а не проверять тут миллион раз
        if(_size != other._size){
            throw std::runtime_error("Tensor sizes must be the same!");
        }
    #endif
    log("Unary Multiplication (", _rank, ")");
    if(_rank == 0/*isScalar()*/){
        log("scalar: ", _value, "*=", other._value);
        _value *= other._value;
    }
    else if(_rank == 1){ // in order to optimize recursive calls overhead
        log("vector: *= ");
        for(int i = 0; i < _size; i++) {
            _coeffs[i]._value *= other._coeffs[i]._value;
        }
    }
    else {
        for(int i = 0; i < _size; i++) {
            // log(_coeffs[i], "*=", other._coeffs[i]); // creates copies
            _coeffs[i] *= other._coeffs[i];
        }
    }
    return *this;
}

// Binary Multiplication Operator
template <Arithmetic U>
/*friend*/ Tensor<U> operator * (const Tensor<U>& t1, const Tensor<U>& t2){
    Tensor<U> temp = t1; // t1 copy in Stack memory allocation
    temp *= t2; // умножаем t2 прямо на temp
    // Return Value Optimization (RVO):
    // problem is since temp is in Stack memory it should be deleted after this function is finished
    return temp;
}
