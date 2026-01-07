/*
    SPDX-License-Identifier: MPL-2.0
    --------------------------------
    This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
    If a copy of the MPL was not distributed with this file,
    You can obtain one at https://mozilla.org/MPL/2.0/.

    This file is part of the Tensor-library:
    https://github.com/alarxx/Tensor-library

    Provided “as is”, without warranty of any kind.

    Copyright © 2026 Alar Akilbekov. All rights reserved.
 */

#include <iostream>
#include <cassert>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <concepts>
#include <initializer_list>
#include <vector>

#include "Tensor/Tensor.hpp"

// using namespace tensor;
using   tensor::Tensor,
        tensor::scalar;

int main(){

    Tensor sc1 = scalar(42.0);
    Tensor sc2 = scalar(45.0);
    sc1.get() = sc2.get();
    sc1.get() += 1;
    std::cout << "scalar: " << sc1.get() << std::endl; // 46
    std::cout << "scalar: " << sc2.get() << std::endl; // 45

    int dims[]{3, 3, 3};
    Tensor<float> t(3, dims);

    /*
    Tensor t(3, 3, 3);
    std::cout << t << std::endl;
    */

    /*
    t.get(1, 1, 1) = 3;
    std::cout << "t.get(1, 1, 1): " << t.get(1, 1, 1) << std::endl;
    */

    /*
    // --- initializer_list ---
    Tensor vec = {1, 2, 3};
    std::cout << "vec.get(1): " << vec.get(1) << std::endl;

    Tensor mat = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
    };
    std::cout << "mat.get(1, 1): " << mat.get(1, 1) << std::endl;

    Tensor d3 = {
        {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        },
        {
            {10, 11, 12},
            {10, 11, 12},
            {10, 11, 12}
        },
    };

    Tensor d4 = { // 4D
        {{ // 3D
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        },
        {
            {10, 11, 12},
            {10, 11, 12},
            {10, 11, 12}
        }},
        {{ // 3D
            {13, 14, 15},
            {16, 17, 18},
            {19, 20, 21},
        },
        {
            {22, 23, 24},
            {25, 26, 27},
            {28, 29, 30}
        }},
    };
    std::cout << "d4: " << d4 << std::endl;
    // ------
    */

    /*
    // #include <vector>
    std::vector<int> stdvec {1, 2, 3};
    std::vector<int> stdvec2 {};
    std::cout << stdvec.size() << std::endl; // 3
    std::cout << stdvec2.size() << std::endl; // 0
    stdvec2 = stdvec;
    std::cout << stdvec2.size() << std::endl; // 3

    Tensor<float> vec = {1, 2, 3};
    // Tensor t2 = vec;
    // Tensor t2 = std::move(vec);
    Tensor<float> t2;
    t2 = vec;
    // t2 = std::move(vec);
    t2.get(0) = 5;
    t2.get(1) = 6;
    t2.get(2) = 7;
    std::cout << "vec " << vec << std::endl;
    std::cout << "t2 " << t2 << std::endl;
    */

}

/*

 - [ ] Full functionality of c++recursive
 - [ ] Slices using index operator []

 */
