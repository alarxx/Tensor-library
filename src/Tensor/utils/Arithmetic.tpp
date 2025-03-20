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

// Requires:
// #include <type_traits>
// #include <concepts>

namespace tensor {

// namespace {
template <typename T> // C++20
concept Arithmetic = requires(T a, T b) {
    /*
        Specialization of std::is_arithmetic for custom class A.
        Может быть интересно, если хотите использовать custom-ный тип в Tensor.
        Если мы включим, то is_arithmetic_v<A> = true, и он будет проходить проверку SFINAE.
        I'd not recommend to do it, actually.

            class A {};

            template <typename T>
            class std::is_arithmetic<A> : public std::true_type {};

        Свой custom-ный тип может быть полезен если хотите Int с проверкой на Overflow.
    */
    // requires std::is_arithmetic_v<T>;

    // specific check for what we need
    { a += b };
    { a -= b };
    { a *= b };
    { a /= b };

    { a + b } -> std::same_as<T>;
    { a - b } -> std::same_as<T>;
    { a * b } -> std::same_as<T>;
    { a / b } -> std::same_as<T>;
};
// }

} // namespace tensor
