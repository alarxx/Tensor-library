/*
    SPDX-License-Identifier: MPL-2.0
    --------------------------------
    This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
    If a copy of the MPL was not distributed with this file,
    You can obtain one at https://mozilla.org/MPL/2.0/.

    Provided “as is”, without warranty of any kind.

    Copyright © 2026 Alar Akilbekov. All rights reserved.
*/

#pragma once
#ifndef _TENSOROPS_
#define _TENSOROPS_

#include "TensorX/Tensor.hpp"

#include <string>
#include <algorithm>
#include <cassert>


namespace tensor {

    template<Arithmetic T>
    bool equal(Tensor<T>& a, Tensor<T>& b){
        // same object
        if (&a == &b) return true;

        // ranks must match
        if (a.getRank() != b.getRank()) return false;

        int rank = a.getRank();
        int* ad = a.getDims();
        int* bd = b.getDims();

        // dims must match (handle null dims safely)
        if (rank > 0){
            if (ad == nullptr || bd == nullptr) return false;
            for (int i = 0; i < rank; ++i){
                if (ad[i] != bd[i]) return false;
            }
        }

        int n = a.getLength(); // b.getLength()
        // Directly use coeffs
        T * ac = a.getCoeffs();
        T * bc = b.getCoeffs();

        if (n > 0 && (ac == nullptr || bc == nullptr)) return false;

        for (int i = 0; i < n; ++i){
            if (ac[i] != bc[i]) return false;
        }
        return true;
    }


    template<Arithmetic T, typename Op>
    Tensor<T> elementwise(
        Tensor<T>& a, Tensor<T>& b,
        Op op, std::string opname = "elementwise"
    ){
        if (a.getRank() != b.getRank()){
            throw std::runtime_error(opname + ": ranks don't match");
        }

        int rank = a.getRank();
        int* ad = a.getDims();
        int* bd = b.getDims();
        // dims must match
        if (rank > 0){
            if (ad == nullptr || bd == nullptr){
                throw std::runtime_error(opname + ": nullptr dims");
            }
            for (int i = 0; i < rank; ++i){
                if (ad[i] != bd[i]){
                    throw std::runtime_error(opname + ": dims don't match");
                }
            }
        }

        int n = a.getLength(); // b.getLength()
        // Directly use coeffs
        T * ac = a.getCoeffs();
        T * bc = b.getCoeffs();

        if (n > 0 && (ac == nullptr || bc == nullptr)){
            throw std::runtime_error(opname + ": nullptr coeffs");
        }

        Tensor<T> out(rank, ad);
        T * oc = out.getCoeffs();
        for (int i = 0; i < n; ++i){
            oc[i] = op(ac[i], bc[i]);
        }
        return out;
    }

    // Scalar
    template<Arithmetic T, typename Op>
    Tensor<T> elementwise(
        T sc, Tensor<T>& t,
        Op op, std::string opname = "elementwise"
    ){
        // Directly use coeffs
        T * tc = t.getCoeffs();

        Tensor<T> out(t.getRank(), t.getDims());
        T * oc = out.getCoeffs();

        for (int i = 0; i < t.getLength(); ++i){
            oc[i] = op(sc, tc[i]);
        }
        return out;
    }


    template<Arithmetic T>
    Tensor<T> mul(Tensor<T>& a, Tensor<T>& b){
        return elementwise<T>(a, b, [](T x, T y){ return x * y; }, "mul");
    }
    template<Arithmetic T>
    Tensor<T> div(Tensor<T>& a, Tensor<T>& b){
        return elementwise<T>(a, b, [](T x, T y){ return x / y; }, "div");
    }
    template<Arithmetic T>
    Tensor<T> add(Tensor<T>& a, Tensor<T>& b){
        return elementwise<T>(a, b, [](T x, T y){ return x + y; }, "add");
    }
    template<Arithmetic T>
    Tensor<T> sub(Tensor<T>& a, Tensor<T>& b){
        return elementwise<T>(a, b, [](T x, T y){ return x - y; }, "sub");
    }

    // Scalar
    template<Arithmetic T>
    Tensor<T> mul(T sc, Tensor<T>& t){
        return elementwise<T>(sc, t, [](T sc, T v){ return sc * v; }, "mul");
    }
    template<Arithmetic T>
    Tensor<T> div(T sc, Tensor<T>& t){
        return elementwise<T>(sc, t, [](T sc, T v){ return v / sc; }, "div");
    }
    template<Arithmetic T>
    Tensor<T> add(T sc, Tensor<T>& t){
        return elementwise<T>(sc, t, [](T sc, T v){ return v + sc; }, "add");
    }
    template<Arithmetic T>
    Tensor<T> sub(T sc, Tensor<T>& t){
        return elementwise<T>(sc, t, [](T sc, T v){ return v - sc; }, "sub");
    }


    template<Arithmetic T>
    T find_max(int length, T * values){
        T maxv = (T) INT_MIN;
        for(int i = 0; i < length; ++i){
            if(maxv < values[i]){
                maxv = values[i];
            }
        }
        return maxv;
    }

    template<Arithmetic T>
    T find_median(int length, T * values){
        std::vector<T> vec(values, values + length);
        std::sort(vec.begin(), vec.end());
        return vec[length / 2];
    }

    template<Arithmetic T>
    T find_median(Tensor<T> t){
        return find_median(t.getLength(), t.getCoeffs());
    }


    template<Arithmetic T>
    T find_mean(int length, T * values){
        T sum = (T) 0;
        for (int i = 0; i < length; ++i) {
            sum += values[i];
        }
        return sum / length;
    }

    template<Arithmetic T>
    T find_mean(Tensor<T> t){
        return find_mean(t.getLength(), t.getCoeffs());
    }


    template<Arithmetic T, Arithmetic S>
    void fill(S value, Tensor<T>& t){
        T * tc = t.getCoeffs();
        for(int i = 0; i < t.getLength(); ++i){
            tc[i] = (T) value;
        }
    }

};

// Tests
namespace tensor {

    void test_mul(){
        Tensor<float> a = {
            {1, 2, 3},
            {3, 4, 5}
        };
        Tensor<float> b = {
            {5, 6, 7},
            {6, 7, 8}
        };
        Tensor<float> expected = {
            { 5, 12, 21},
            {18, 28, 40}
        };

        Tensor<float> out = mul(a, b);
        std::cout << out << std::endl;
        assert(equal(out, expected) && "mul result is not correct");
    }

    void test_mul_scalars(){
        Tensor a = tensor::scalar(6.0f);
        Tensor b = tensor::scalar(7.0f);
        Tensor expected = tensor::scalar(42.0f);
        Tensor out = mul(a, b);
        std::cout << out << std::endl;
        assert(equal(out, expected) && "mul result is not correct");
    }

    void test_scalar_mul(){
        Tensor<float> a = {
            {1, 2, 3},
            {3, 4, 5}
        };
        Tensor<float> expected = {
            {2,  4,  6},
            {6,  8, 10}
        };

        Tensor out = mul(2.0f, a);
        std::cout << out << std::endl;
        assert(equal(out, expected) && "mul result is not correct");
    }

};

#endif
