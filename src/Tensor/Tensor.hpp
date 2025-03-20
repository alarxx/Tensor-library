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
    // Tensor<T> * _coeffs;
    T * _coeffs;
    int * _shape; // _rank = _shape.length

public:
    explicit Tensor() : _value(0), _rank(0), _size(-1), _coeffs(nullptr), _shape(nullptr) {}

    explicit Tensor(std::integral auto ... args){
        log("Basic Constructor");

        int dims[] = {args...}; // arguments expansion
        int length = sizeof...(args);

        if(length <= 0){ // never less actually
            throw std::runtime_error("Error: empty dims!");
        }

        _rank = length;
        _shape = new int[length]; // dims is in Stack memory
        for(int i = 0; i < length; i++){
            _shape[i] = dims[0];
        }


        // this constructor is never scalar, at least vector of size 1 * dims[0]
        _size = 1;
        for(int i = 0; i < _rank; i++){
            _size *= _shape[i];
        }

        log("_rank: ", _rank, "; _size: ", _size);

        _coeffs = new T[_size];
    }

    // // Copy Assignment Operator (Scalar)
    // // tensor[index] = number;
    // Tensor<T> & operator = (const T & scalar);
    // // Index Operator []

    inline T get(std::integral auto ... args){
        if(isScalar()){
            return _value;
        }
        int dims[] = {args...}; // arguments expansion
        int length = sizeof...(args);
        if(length != _rank){
            throw std::runtime_error("Error: invalid indexes!");
        }
        return _coeffs[index(dims)];
    }

    inline Tensor<T> & set(T value, std::integral auto ... args){
        if(isScalar()){
            _value = value;
        }
        else {
            int dims[] = {args...}; // arguments expansion
            int length = sizeof...(args);
            if(length != _rank){
                throw std::runtime_error("Error: invalid indexes!");
            }
            _coeffs[index(dims)] = value;
        }
        return *this;
    }

    inline bool isScalar() const { return _rank == 0 && _size == -1 && _coeffs == nullptr && _shape == nullptr; }
    inline bool isVector() const { return _rank == 1 && _size != -1 && _coeffs != nullptr && _shape != nullptr; }
    inline bool isMatrix() const { return _rank == 2 && _size != -1 && _coeffs != nullptr && _shape != nullptr; }

    inline int size() const { return _size; }
    inline int rank() const { return _rank; }
    inline int * shape() const { return _shape; }

    inline T& value(){
        #if DEBUG_TENSOR
            if(!isScalar()){
                throw std::runtime_error("Error: Geting a value of a non-scalar tensor!");
            }
        #endif
        return _value;
    }

    inline int index(const int * const dims) const {
        int index = 0;
        if(dims[0] >= _shape[0]){
            throw std::runtime_error("Error: out of bounds!");
        }
        for(int i = 0; i < _rank - 1; i++){
            index += _shape[i + 1] * dims[i];
        }
        index += dims[_rank - 1]; // last element
        return index;
    }

    std::string toString(){
        if(isScalar()){
            return "tensor<" + std::string(typeid(T).name()) + ">: " + std::to_string(_value);
        }

        std::string result = "tensor<" + std::string(typeid(T).name()) + ">: \n";

        for(int i = 0; i < _size; i++){
            result += std::string(" ") + std::to_string(_coeffs[i]);
            if((i + 1) % _shape[_rank - 1] == 0){
                result += "\n";
            }
            if(_rank > 2){
                if((i + 1) % (_shape[_rank - 1] * _shape[_rank - 2]) == 0){
                    result += "\n";
                }
            }
        }
        if(_rank > 2) result.pop_back(); // last '\n'
        return result;
    }

};

} // namespace tensor

/*
    Implementation of template class is in .tpp file
 */

#endif
