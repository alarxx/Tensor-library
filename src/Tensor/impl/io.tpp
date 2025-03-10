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

// --- toString ---

template <Arithmetic T>
std::string Tensor<T>::__toString() const {
    // log("rank: ", _rank); // 3 2 1 0 0 0 3 2 1 0 0 0
    std::string res = "";
    if(_rank == 0){
        res += std::to_string(_value) + std::string(typeid(_value).name()) + " ";
    }
    else { // _rank != 0
        for(int i = 0; i < _size; i++){
            res += _coeffs[i].__toString();
        }
        if(_rank <= 2){ // vector and matrix oriented output
            res += "\n";
        }
    }
    return res;
}

template <Arithmetic T>
std::string Tensor<T>::toString() const {
    std::string res = "";
    res += "tensor(" + std::to_string(_rank) + ")<" + std::string(typeid(_value).name()) + ">:";
    if(!isScalar()){
        res += "{\n";
        res += __toString();
        if(_rank >= 2) res.pop_back(); // last '\n'
        res += "}";
    }
    else {
        res += " " + std::to_string(_value) + std::string(typeid(_value).name());
    }
    return res;
}

// ------


// --- Operator Overloadings ---

// Stream insertion operation
template <Arithmetic U>
/*friend*/ std::ostream& operator << (std::ostream& os, const Tensor<U>& tensor){
    os << tensor.toString();
    return os;
}

// ------

} // namespace tensor
