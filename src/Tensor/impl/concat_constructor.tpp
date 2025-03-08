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

// Copy concat constructor
template <Arithmetic T>
template <typename ... TArgs>
requires (std::is_same_v<std::remove_reference_t<TArgs>, Tensor<T>> && ... && true)
Tensor<T>::Tensor(const Tensor<T>& first, const TArgs& ... args) {
    log("Concat constructor (&)");
    Tensor<T> tensors[] = {first, args...}; // creates copy (Ok)
    // Tensor<T> tmp = {first, args...}; // Concat {tensors} creates copy (Bad)

    _size = sizeof(tensors) / sizeof(tensors[0]);
    _rank = tensors[0]._rank + 1;

    _coeffs = new Tensor<T>[_size];

    for(int i = 0; i < _size; i++) {
        _coeffs[i]._rank = tensors[i]._rank;
        _coeffs[i]._size = tensors[i]._size;
        _coeffs[i] = std::move(tensors[i]); // move assignment
    }
}

// Move concat constructor
template <Arithmetic T>
template <typename ... TArgs>
requires (std::is_same_v<std::remove_reference_t<TArgs>, Tensor<T>> && ... && true)
Tensor<T>::Tensor(Tensor<T>&& first, TArgs&& ... args) {
    log("Concat constructor (&&)");
    // Tensor<T> tensors[] = {first, args...}; // creates copy
    // Tensor<T> tmp = {first, args...}; // Concat {tensors} creates copy 2 times (Bad)

    std::vector<Tensor<T>> tensors;

    int size = sizeof...(args) + 1;
    log("size: ", size);

    assert(size - 1 > 0 && "Error: With one argument it should be a move constructor, not a concat constructor!");

    tensors.reserve(size);

    /*
        emplace_back(Args && ... args) vs. push_back(T & obj)

        Мне кажется будто нет разницы между этими методами.

        Если мы передаем через std::move результат будет одинковым у emplace_back и push_back.
        Разница будет только если мы захотим создать объект по аргументам прямо через них.

            emplace_back(arg1, arg2);
            push_back(Object(arg1, arg2));

     */
    // tensors.push_back(std::move(first)); // lvalue& =
    tensors.emplace_back(std::move(first)); // rvalues are arguments, so it allocates Tensor(std::move(first))
    // https://en.cppreference.com/w/cpp/language/fold
    (tensors.emplace_back(std::move(args)), ...);

    _size = size;
    _rank = tensors[0]._rank + 1;

    _coeffs = new Tensor<T>[_size];

    for(int i = 0; i < _size; i++) {
        _coeffs[i]._rank = tensors[i]._rank;
        _coeffs[i]._size = tensors[i]._size;
        _coeffs[i] = std::move(tensors[i]); // move assignment
    }

}
