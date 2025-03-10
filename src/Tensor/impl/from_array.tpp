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

// --- Tensor from array ---

template <typename U>
requires Arithmetic<U>
/*private friend*/ Tensor<U> _from_array(
    const std::vector<int>& dims,
    void * arr,
    const long unsigned int cursor = 0
){
    const int size = dims.begin()[cursor]; // *(dims.begin() + cursor)

    log("Tensor from array: size=", size, ", arr=", arr, " type=", typeid(arr).name());

    Tensor<U> tensor(size); // as a vector
    tensor._rank = (int) dims.size() - cursor;

    if(tensor._rank == 1){ // last - scalar values
        log("array of primitive types (size=", size, ")");
        for(int i = 0; i < size; i++){
            // we can't access initializer_list by index, only by range-based for loop or by iterator
            // log("loop(", i+1, "/", size, "): ", arr[i], ", type=", typeid(arr[i]).name());
            log("loop(", i+1, "/", size, "): ");
            tensor._coeffs[i] = static_cast<U*>(arr)[i];
        }
    }
    else {
        for(int i = 0; i < size; i++){
            // we can't access initializer_list by index, only by range-based for loop or by iterator
            // log("loop(", i+1, "/", size, "): ", arr[i], ", type=", typeid(arr[i]).name());
            log("loop(", i+1, "/", size, "): ");
            // Tensor& next = tensor._coeffs[i];
            tensor._coeffs[i]._rank = tensor._rank - 1;
            tensor._coeffs[i]._size = dims.begin()[cursor + 1lu];
            tensor._coeffs[i] = _from_array<U>(
                    dims,
                    static_cast<void*>(
                        static_cast<U*>(arr) + i * tensor._coeffs[i]._size
                    ),
                    cursor + 1lu
                );
        }
    }
    return tensor;
}

template <typename U, int SIZE>
requires Arithmetic<std::remove_all_extents_t<U>>
/*friend*/ Tensor<std::remove_all_extents_t<U>> from_array(
    const std::vector<int>& dims,
    U (&arr)[SIZE]
){
    if(dims.size() == 0){
        throw std::runtime_error("Error: `dims` are empty! Maybe take a look at the scalar() factory function.");
    }
    log("Initial Tensor from array: SIZE=", SIZE, " array=", arr, " type=", typeid(U).name());
    return _from_array<std::remove_all_extents_t<U>>(dims, static_cast<void*>(arr));
}

// ------

} // namespace tensor
