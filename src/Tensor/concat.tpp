#include "Tensor.hpp"

// Copy concat
template <Arithmetic U, typename ... TArgs>
requires (std::is_same_v<std::remove_reference_t<TArgs>, Tensor<U>> && ... && true)
Tensor<U> concat(const Tensor<U>& first, const TArgs& ... args){
    log("Concat (&)");
    Tensor<U> tensors[] = {first, args...}; // creates copy (Ok)
    // Tensor<U> tmp = {first, args...}; // Concat {tensors} creates copy (Bad)

    int size = sizeof(tensors) / sizeof(tensors[0]);
    int rank = tensors[0]._rank + 1;

    // return concat(size, tensors); // Почему-то через функцию создает копии

    Tensor<U> tmp(size);
    tmp._rank = rank; // friend
    // tmp._size = size; // already there

    for(int i = 0; i < size; i++) {
        tmp._coeffs[i]._rank = tensors[i]._rank;
        tmp._coeffs[i]._size = tensors[i]._size;
        tmp._coeffs[i] = std::move(tensors[i]); // move assignment
    }
    return tmp;
}

// Move concat
template <Arithmetic U, typename ... TArgs>
requires (std::is_same_v<std::remove_reference_t<TArgs>, Tensor<U>> && ... && true)
Tensor<U> concat(const Tensor<U>&& first, const TArgs&& ... args){
    log("Concat (&&)");
    // Tensor<U> tensors[] = {first, args...}; // creates copy (Bad)!!! we must use `emplace_back`

    Tensor<U> tmp = {first, args...}; // Concat {tensors} creates copy (Bad)

    return tmp;
}

template <Arithmetic U>
Tensor<U> concat(const int size, const Tensor<U> tensors[]){ // Copy
    log("Concat (size, Tensor[]))");
    Tensor<U> tmp(size);
    int rank = tensors[0]._rank + 1;
    tmp._rank = rank; // friend
    // tmp._size = size; // already there

    for(int i = 0; i < size; i++) {
        tmp._coeffs[i]._rank = tensors[i]._rank;
        tmp._coeffs[i]._size = tensors[i]._size;
        tmp._coeffs[i] = std::move(tensors[i]); // move assignment
    }
    log("Concat (size, Tensor[])) return");
    return tmp;
}
