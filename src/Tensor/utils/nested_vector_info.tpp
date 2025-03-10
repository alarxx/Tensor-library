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

/*
    nested_vector_info

    inspired by std::remove_all_extents_t, but with vector rather than raw array
        - https://en.cppreference.com/w/cpp/types/remove_all_extents

        template <typename U>
        void print_type(const std::vector<U>& vector){
            std::cout << typeid(typename nested_vector_info<U>::base_type).name() << std::endl; // i
        }

        int main(){
            std::vector<std::vector<int>> matrix = {
                {1, 2},
                {3, 4}
            };
            print_type(matrix);
        }

*/
// Base case
template <typename T>
class nested_vector_info {
public:
    using base_type = T;
    static constexpr int depth = 0;
};


// Recursive case
template <typename T>
class nested_vector_info<std::vector<T>> {
public:
    using base_type = typename nested_vector_info<T>::base_type;
    static constexpr int depth = 1 + nested_vector_info<T>::depth;
};


/*
    Partial specialization for lvalue-references, just "redirects"
    Иначе при передаче ссылки срабатывает base case
    Похоже на Deduction Guide, когда мы указываем тип при определенных параметрах конструктора

    Base case:

        template <typename T> struct A { using type = void; };

    int specialization:

        template <> struct A<int> { using type = int; };

    Это специализация когда передается ссылка, получается мы избавляемся от ссылки:

        // Uncomment it to call int specialization when int& is used
        // Otherwise base case will be called
        template <typename T> struct A<T&> : public A<T> {};

    Move reference specialization:

        template <typename T> struct A<T&&> : public A<T& or just T> {};

    Usage:

        int main(){
            A<int> a;
            std::cout << typeid(decltype(a)::type).name() << std::endl; // i

            A<int&> aref;
            std::cout << typeid(decltype(aref)::type).name() << std::endl; // i or v

            A<float&> fref;
            std::cout << typeid(decltype(fref)::type).name() << std::endl; // v
        }
 */
template <typename T>
class nested_vector_info<T&> : public nested_vector_info<T> {};


// Partial specialization for rvalue-references
template <typename T>
class nested_vector_info<T&&> : public nested_vector_info<T> {};


// Alias for short
template <typename T>
using nested_vector_info_t = typename nested_vector_info<T>::base_type;

} // namespace tensor
