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


/*
    is_vector

        std::vector<int> va = {1, 2, 3};
        std::cout << is_vector_v<decltype(va)> << std::endl; // 1

        int arr[3] = {1, 2, 3};
        std::cout << is_vector_v<decltype(arr)> << std::endl; // 0
 */
template <typename T>
class is_vector : public std::false_type {};

template <typename T>
class is_vector<std::vector<T>> : public std::true_type {};

template <typename T>
constexpr bool is_vector_v = is_vector<T>::value;
