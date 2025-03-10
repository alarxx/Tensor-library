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

// --- initializer_list ---

// Tensor vector = {1, 2, 3};
template <Arithmetic T>
Tensor<T>::Tensor(const INITIALIZER_LIST_1(T) list){
    log("INIT_LIST_1");
    // 1D: {1, 2, 3}
    _rank = 1;
    _size = list.size();
    _coeffs = new Tensor<T>[_size];

    int i = 0;
    for(auto e: list){
        // we can't access initializer_list by index, only by range-based for loop or by iterator
        _coeffs[i++] = e; // copy scalar assignment =
    }
}

// Tensor matrix = {{1, 2}, {3, 4}};
template <Arithmetic T>
Tensor<T>::Tensor(const INITIALIZER_LIST_2(T) list){
    // 2D:
    // {{1, 2, 3},
    //  {4, 5, 6},
    //  {7, 8, 9}}
    _rank = 2;

    log("INIT_LIST_", _rank);

    _size = list.size();
    _coeffs = new Tensor<T>[_size];

    int i = 0;
    for(auto & e: list){
        _coeffs[i]._rank = 1;
        _coeffs[i]._size = e.size();
        _coeffs[i++] = Tensor(e); // Move Assignment =
    }
}

template <Arithmetic T>
Tensor<T>::Tensor(const INITIALIZER_LIST_3(T) list){
     _rank = 3; // После _rank все одинаково, просто рекурсивно вызываются конструкторы меньшего порядка
     log("INIT_LIST_", _rank);
    _size = list.size(); _coeffs = new Tensor<T>[_size];
    int i = 0; for(auto & e: list){ _coeffs[i]._rank = _rank - 1; _coeffs[i]._size = e.size(); _coeffs[i++] = Tensor(e); }
}

template <Arithmetic T>
Tensor<T>::Tensor(const INITIALIZER_LIST_4(T) list){
    _rank = 4;
    log("INIT_LIST_", _rank);
    _size = list.size(); _coeffs = new Tensor<T>[_size];
    int i = 0; for(auto & e: list){ _coeffs[i]._rank = _rank - 1; _coeffs[i]._size = e.size(); _coeffs[i++] = Tensor(e); }
}

// ------

} // namespace tensor
