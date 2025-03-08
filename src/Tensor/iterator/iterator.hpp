#pragma once
#ifndef _ITERATOR_H_
#define _ITERATOR_H_

#include <iterator> // random_access_iterator_tag
#include <cstddef> // ptrdiff_t

/*
    --- iterator ---

    Main article:
        https://www.internalpointers.com/post/writing-custom-iterators-modern-cpp

    iterator_tags:
        https://en.cppreference.com/w/cpp/iterator/iterator_tags
        - Input (read-only)         input_iterator_tag
        - Output (write-only)       output_iterator_tag
        - Forward Iterator          forward_iterator_tag
        - Bidirectional Iterator    bidirectional_iterator_tag
        - Random Access Iterator    random_access_iterator_tag
        - Contiguous Iterator       contiguous_iterator_tag

    The properties of each iterator category are:
        https://cplusplus.com/reference/iterator/

    iterator
    constant_iterator - это по сути Input Iterator (read-only)
    reverse_iterator
    constant_reverse_iterator

    --- Output Iterator ---
    Мы здесь не реализуем чистый Output Iterator.
    Мы же dereferencing делаем так *_ptr,
    а при Ouput Iterator мы делаем dereferencing так *this,
    то есть это фиктивное разыменование, it returns itself,
    то есть возвращаем сам iterator, чтобы потом воспользовать assign operator =:
    *it = 42;

        struct Output {
            T* _ptr;
            Output& operator * (){ return *this; }
            Output& operator = (const T& data){
                *_ptr = data;
            }
        };
    ------
*/
template <typename T>
class iterator {
public:
    // https://en.cppreference.com/w/cpp/iterator/iterator_traits
    // https://en.cppreference.com/w/cpp/iterator/iterator_tags
    using iterator_category = std::random_access_iterator_tag;
    // https://en.cppreference.com/w/cpp/types/ptrdiff_t
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T*; // value_type*
    using reference = T&; // value_type&
private:
    pointer _ptr;
public:
    // constructible, copy-constructible, copy-assignable, destructible, and swappable
    iterator(pointer ptr) : _ptr(ptr) {}

    reference operator * (){ return *_ptr; }
    pointer operator -> (){ return _ptr; }

    // prefix increment
    iterator& operator ++ (){ _ptr++; return *this; }
    // postfix increment
    iterator operator ++ (int){
        iterator tmp = *this; // overhead on creation of temporary object
        ++(*this);
        return tmp;
    }

    // prefix decrement
    iterator& operator -- (){ _ptr--; return *this; }
    // postfix decrement
    iterator operator -- (int){
        iterator tmp = *this; // overhead on creation of temporary object
        --(*this);
        return tmp;
    }

    iterator operator + (difference_type n){ return iterator(_ptr + n); }
    iterator operator - (difference_type n){ return iterator(_ptr - n); }

    difference_type operator - (const iterator& other){ return _ptr - other._ptr; }

    reference operator [] (difference_type index){ return _ptr[index]; }

    friend bool operator == (const iterator& a, const iterator& b){ return a._ptr == b._ptr; }
    friend bool operator != (const iterator& a, const iterator& b){ return a._ptr != b._ptr; }
    friend bool operator <  (const iterator& a, const iterator& b){ return a._ptr <  b._ptr; }
    friend bool operator >  (const iterator& a, const iterator& b){ return a._ptr >  b._ptr; }
    friend bool operator <= (const iterator& a, const iterator& b){ return a._ptr <= b._ptr; }
    friend bool operator >= (const iterator& a, const iterator& b){ return a._ptr >= b._ptr; }
};


/*

We can just use std::reverse_iterator<iterator>,
so no need in explicitly defining the reverse_iterator class

// --- reverse_iterator ---
template <typename T>
class reverse_iterator {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T*; // value_type*
    using reference = T&; // value_type&
private:
    pointer _ptr;
public:
    reverse_iterator(pointer ptr) : _ptr(ptr) {}

    reference operator * (){ return *_ptr; }
    pointer operator -> (){ return _ptr; }
    // prefix increment
    reverse_iterator& operator ++ (){ _ptr--; return *this; }
    // postfix increment
    reverse_iterator operator ++ (int){
        reverse_iterator tmp = *this; // overhead on creation of temporary object
        --(*this);
        return tmp;
    }

    friend bool operator == (const reverse_iterator& a, const reverse_iterator& b){
        return a._ptr == b._ptr;
    }
    friend bool operator != (const reverse_iterator& a, const reverse_iterator& b){
        return a._ptr != b._ptr;
    }
};
*/

#endif
