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

#include "Tensor.hpp"


// --- Logging ---
template <typename T>
concept __is_stream_supported = requires (T t){
    // SFINAE constrains
    std::declval<std::ostream&>() << t;
};

template <typename T>
requires __is_stream_supported<T>
void log(T t){
    #if DEBUG_LOG_TENSOR
        std::cout << t << std::endl;
    #endif
}

template <typename T, typename ... TArgs>
requires __is_stream_supported<T>
void log(T t, TArgs ... args){
    #if DEBUG_LOG_TENSOR
        std::cout << t;
        log(args...);
    #endif
}
// ------


// Constructor : Tensor tensor(depth, rows, cols)
template <Arithmetic T>
Tensor<T>::Tensor(std::integral auto ... args) { // C++20, abbreviated function templates with concept
    int dims[] = {args...}; // arguments expansion

    if(dims[0] <= 0){
        throw std::runtime_error("Error: Null tensor, dims[0] is invalid!");
    }

    int rank = sizeof(dims) == 0 ? 0 : sizeof(dims) / sizeof(dims[0]);

    __init(rank, dims, 0);
}


template <Arithmetic T>
/*private*/ Tensor<T>::Tensor(int rank, int dims[], int cursor) {
    if(rank < 0){
        throw std::runtime_error("Error: Null tensor, rank is invalid!");
    }
    __init(rank, dims, cursor);
}


template <Arithmetic T>
/*private*/ inline void Tensor<T>::__init(int rank, int dims[], int cursor){
    // 0D: (0, {})
    // 1D: (1, {4})
    // 2D: (2, {4, 4})
    // 3D: (3, {4, 4, 4})
    log(cursor, ") rank-", rank, " Constructor of Tensor");

    _rank = rank;
    _size = dims[cursor];
    if(_rank < 0){
        throw std::runtime_error("Error: rank is invalid!");
    }
    else if(_size <= 0){
        throw std::runtime_error("Error: size is invalid!");
    }

    // --- recursively initialization of nested tensors ---
    _coeffs = new Tensor<T>[_size];
    // after this, tensors look like scalars, but it's okay we'll move rvalues into them

    if(rank - 1 == 0) { // then it is a Scalar
        log("Next is scalar tensor, _size:(", _size, "), so we return");
        return;
    }

    for(int i = 0; i < _size; i++){
        // Чтобы он не выглядел как scalar, мы назначаем _rank и _size
        _coeffs[i]._rank = rank - 1;
        _coeffs[i]._size = dims[cursor + 1];
        _coeffs[i] = Tensor(rank - 1, dims, cursor + 1); // Move Assignment =
    }
    // ------
}

// --- Tensor from array ---

template <typename U>
requires Arithmetic<U>
/*private friend*/ Tensor<U> _from_array(
    const std::vector<int> dims,
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
    const std::vector<int> dims,
    U (&arr)[SIZE]
){
    if(dims.size() == 0){
        throw std::runtime_error("Error: `dims` are empty! Maybe take a look at the scalar() factory function.");
    }
    log("Initial Tensor from array: SIZE=", SIZE, " array=", arr, " type=", typeid(U).name());
    return _from_array<std::remove_all_extents_t<U>>(dims, static_cast<void*>(arr));
}

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


// ------

// Tensor sc = scalar(42.0);
template <Arithmetic U>
/*friend*/ Tensor<U> scalar(U value){
    Tensor<U> tensor;
    tensor = value;
    // RVO
    return tensor;
}

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
    int i = 0; for(auto & e: list){ _coeffs[i]._rank = 1; _coeffs[i]._size = e.size(); _coeffs[i++] = Tensor(e); }
}

template <Arithmetic T>
Tensor<T>::Tensor(const INITIALIZER_LIST_4(T) list){
    _rank = 4;
    log("INIT_LIST_", _rank);
    _size = list.size(); _coeffs = new Tensor<T>[_size];
    int i = 0; for(auto & e: list){ _coeffs[i]._rank = 1; _coeffs[i]._size = e.size(); _coeffs[i++] = Tensor(e); }
}

// ------

// --- Rule of 5 ---

template <Arithmetic T>
Tensor<T>::~Tensor(){
    log("~ rank-", _rank, " Destructor of Tensor", (isScalar() ? " (Scalar)" : ""));
    if(_coeffs != nullptr){
        log("\tdelete[] coeffs");
        // рекурсивно удаляются nested tensors
        delete[] _coeffs;
        _coeffs = nullptr;
    }
}


// Copy Constructor
// O(N)
template <Arithmetic T>
Tensor<T>::Tensor(const Tensor<T> & other) : Tensor() {
    log("Copy Constructor", (other.isScalar() ? " (Scalar)" : ""), " (rank=", other._rank, ", size=", other._size, ")");

    _rank = other._rank;
    _size = other._size;
    // Мы можем назначать и scalar tensor, не только используя .value() method to assign scalar
    _value = other._value;

    if(_rank > 0){
        _coeffs = new Tensor<T>[_size];

        for(int i = 0; i < _size; i++){
            // Чтобы он не выглядел как scalar, мы назначаем _rank и _size
            _coeffs[i]._rank = other._coeffs[i]._rank;
            _coeffs[i]._size = other._coeffs[i]._size;
            _coeffs[i] = other._coeffs[i]; // Copy Assignment =
        }
    }
}

// Copy Assignment Operator
// O(N)
template <Arithmetic T>
Tensor<T> & Tensor<T>::operator = (const Tensor<T> & other){
    log("Copy Assignment Operator", (isScalar() ? " (Scalar)" + std::to_string(other._value) : ""), " (rank=", other._rank, "->", _rank, ", size=", other._size, "->", _size, ")");

    if(this != &other){

        if (_rank != other._rank){
            throw std::runtime_error("Error: rank doesn't match for copy assignment");
        }
        if(_size != other._size){
            throw std::runtime_error("Error: size doesn't match for copy assignment");
        }

        if(_coeffs != nullptr){
            log("\tdelete[] coeffs");
            delete[] _coeffs;
            _coeffs = nullptr;
        }

        // _rank и _size уже одинаковые
        // _rank = other._rank;
        // _size = other._size;
        // Мы можем назначать и scalar tensor, не только используя .value() method to assign scalar
        _value = other._value;

        if(_rank > 0){ // if _rank == 0 tensor is scalar
            _coeffs = new Tensor<T>[_size];

            for(int i = 0; i < _size; i++){
                // Чтобы он не выглядел как scalar, мы назначаем _rank и _size
                _coeffs[i]._rank = other._coeffs[i]._rank;
                _coeffs[i]._size = other._coeffs[i]._size;
                _coeffs[i] = other._coeffs[i]; // Copy Assignment =
            }
        }

        // no need to delete other
    }

    return *this;
}

// Move Constructor
// O(1)
template <Arithmetic T>
Tensor<T>::Tensor(Tensor<T> && other){
    log("Move Constructor", (isScalar() ? " (Scalar)" : ""), " (rank=", other._rank, ", size=", other._size, ")");
    // this = other
    _value = other._value;
    _rank = other._rank;
    _size = other._size;
    _coeffs = other._coeffs;
    // other = null (looks like scalar)
    other._value = 0;
    other._rank = 0;
    other._size = -1;
    other._coeffs = nullptr;
}


// Move Assignment Operator
// O(1)
template <Arithmetic T>
Tensor<T> & Tensor<T>::operator = (Tensor<T> && other){
    log("Move Assignment Operator", (isScalar() ? " (Scalar)" : ""), " (rank=", other._rank, "->", _rank, ", size=", other._size, "->", _size, ")");

    if(this != &other){

        if (_rank != other._rank){
            throw std::runtime_error("Error: rank doesn't match for move assignment");
        }
        if(_size != other._size){
            throw std::runtime_error("Error: size doesn't match for move assignment");
        }

        if(_coeffs != nullptr){
            log("\tdelete[] coeffs");
            delete[] _coeffs;
            _coeffs = nullptr;
        }

        // _rank и _size уже одинаковые
        // _rank = other._rank;
        // _size = other._size;
        // Мы можем назначать и scalar tensor, не только используя .value() method to assign scalar
        _value = other._value;
        _coeffs = other._coeffs;

        // reset but not delete, будет выглядеть как scalar
        other._value = 0;
        other._rank = 0;
        other._size = -1;
        other._coeffs = nullptr;

    }

    return *this;
}


// Copy Scalar Assignment Operator
// O(1)
template <Arithmetic T>
Tensor<T> & Tensor<T>::operator = (const T & scalar){
    if(!isScalar()){
        throw std::runtime_error("Error: Can't assign scalar to a non-scalar tensor!");
    }
    log("rank-", _rank, " Copy Scalar Assignment Operator: ", scalar, typeid(T).name());
    _value = scalar;
    return *this;
}

// ------

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


// --- Operator Overloadings ---

// Stream insertion operation
template <Arithmetic U>
/*friend*/ std::ostream& operator << (std::ostream& os, const Tensor<U>& tensor){
    os << tensor.toString();
    return os;
}

// Unary Operator
// Elementwise: v1 *= v2;
template <Arithmetic T>
Tensor<T>& Tensor<T>::operator *= (const Tensor<T>& other){
    #if DEBUG_TENSOR
        // Different sizes may cause overflow, нужно ли делать эту проверку
        // Я думаю, лучшим решением будет user-у просто перед умножением проверять размеры, а не проверять тут миллион раз
        if(_size != other._size){
            throw std::runtime_error("Tensor sizes must be the same!");
        }
    #endif
    log("Unary Multiplication (", _rank, ")");
    if(_rank == 0/*isScalar()*/){
        log("scalar: ", _value, "*=", other._value);
        _value *= other._value;
    }
    else if(_rank == 1){ // in order to optimize recursive calls overhead
        log("vector: *= ");
        for(int i = 0; i < _size; i++) {
            _coeffs[i]._value *= other._coeffs[i]._value;
        }
    }
    else {
        for(int i = 0; i < _size; i++) {
            // log(_coeffs[i], "*=", other._coeffs[i]); // creates copies
            _coeffs[i] *= other._coeffs[i];
        }
    }
    return *this;
}

// Binary Multiplication Operator
template <Arithmetic U>
/*friend*/ Tensor<U> operator * (const Tensor<U>& t1, const Tensor<U>& t2){
    Tensor<U> temp = t1; // t1 copy in Stack memory allocation
    temp *= t2; // умножаем t2 прямо на temp
    // Return Value Optimization (RVO):
    // problem is since temp is in Stack memory it should be deleted after this function is finished
    return temp;
}

// ------

