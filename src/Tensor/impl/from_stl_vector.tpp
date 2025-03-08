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

// Tensor from vector
// Получается, будут создаваться разные функции под разное количество вложенности std::vector-ов, но их size расчитывается in-runtime, например.
template <typename U>
requires Arithmetic<nested_vector_info_t<U>>
Tensor<nested_vector_info_t<U>>
from_stl_vector(
    // size хранится в std::vector
    // rank расчитывается рекурсивно в nested_vector_info, как base type расчитывается в remove_all_extents_t
    // если мы посчитаем rank рекурсивно, то можно не передавать dims
    std::vector<U>& vector
){
    using base_type = typename nested_vector_info<decltype(vector)>::base_type;
    constexpr int depth = nested_vector_info<decltype(vector)>::depth;

    const int size = vector.size();

    log("Tensor from STL vector<", typeid(base_type).name(), ">: size=", size, " depth=", depth, " type=", typeid(vector).name());

    Tensor<nested_vector_info_t<U>> tensor(size); // as a vector
    tensor._rank = depth;

    // when the recursion reaches a vector whose element type is not a vector (e.g., int) the function call is invalid
    if constexpr (depth == 1){ // last - scalar values
        log("array of primitive types (size=", size, ")");
        for(int i = 0; i < size; i++){
            log("loop(", i+1, "/", size, "): ");
            tensor._coeffs[i] = vector[i];
        }
    }
    else {
        for(int i = 0; i < size; i++){
            log("loop(", i+1, "/", size, "): ");
            // Tensor& next = tensor._coeffs[i];
            tensor._coeffs[i]._rank = depth - 1;
            tensor._coeffs[i]._size = vector[i].size(); // sizes must be the same, actually
            tensor._coeffs[i] = from_stl_vector(vector[i]);
        }
    }
    return tensor;
}
