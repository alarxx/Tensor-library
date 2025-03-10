#pragma once
#ifndef _CONSTANT_ITERATOR_H_
#define _CONSTANT_ITERATOR_H_

#include <iterator> // random_access_iterator_tag
#include <cstddef> // ptrdiff_t

namespace tensor {

/*
    --- constant_iterator ---

    https://stackoverflow.com/questions/309581/what-is-the-difference-between-const-iterator-and-non-const-iterator-in-the-c

    T*          // A non-const iterator to a non-const element. ≡ vector<T>::iterator
    T* const    // A const iterator to a non-const element. ≡ const vector<T>::iterator
    const T*    // A non-const iterator to a const element. ≡ vector<T>::const_iterator
 */
template <typename T>
class constant_iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = const T*; // value_type*
    using reference = const T&; // value_type&
private:
    pointer _ptr;
public:
    constant_iterator(pointer ptr) : _ptr(ptr) {}

    reference operator * () const { return *_ptr; }
    pointer operator -> () const { return _ptr; }
    // prefix increment
    constant_iterator& operator ++ (){ _ptr++; return *this; }
    // postfix increment
    constant_iterator operator ++ (int){
        constant_iterator tmp = *this; // overhead on creation of temporary object
        ++(*this);
        return tmp;
    }

    // prefix decrement
    constant_iterator& operator -- (){ _ptr--; return *this; }
    // postfix decrement
    constant_iterator operator -- (int){
        constant_iterator tmp = *this; // overhead on creation of temporary object
        --(*this);
        return tmp;
    }

    constant_iterator operator + (difference_type n){ return constant_iterator(_ptr + n); }
    constant_iterator operator - (difference_type n){ return constant_iterator(_ptr - n); }

    difference_type operator - (const constant_iterator& other){ return _ptr - other._ptr; }

    reference operator [] (difference_type index){ return _ptr[index]; }

    friend bool operator == (const constant_iterator& a, const constant_iterator& b){ return a._ptr == b._ptr; }
    friend bool operator != (const constant_iterator& a, const constant_iterator& b){ return a._ptr != b._ptr; }
    friend bool operator <  (const constant_iterator& a, const constant_iterator& b){ return a._ptr <  b._ptr; }
    friend bool operator >  (const constant_iterator& a, const constant_iterator& b){ return a._ptr >  b._ptr; }
    friend bool operator <= (const constant_iterator& a, const constant_iterator& b){ return a._ptr <= b._ptr; }
    friend bool operator >= (const constant_iterator& a, const constant_iterator& b){ return a._ptr >= b._ptr; }
};

/*
// --- constant_reverse_iterator ---
template <typename T>
class constant_reverse_iterator {...}
*/

#endif

} // namespace tensor
