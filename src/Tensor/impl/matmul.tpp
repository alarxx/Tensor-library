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

template <Arithmetic U>
/*friend*/ Tensor<U> matmul(/*const*/ Tensor<U>& left, /*const*/ Tensor<U>& right){
    #if DEBUG_TENSOR
        if(left._rank != right._rank){
            throw std::runtime_error("MatMul Error: Rank must be the same!");
        }
        if(left._rank < 2){
            throw std::runtime_error("MatMul Error: Tensor must be at least matrix!");
        }
    #endif

    log("Matrix multiplication (", left._rank, ")");

    if(left._rank > 2){
        if(left._size != right._size){
            #if DEBUG_TENSOR
                throw std::runtime_error("MatMul Error: Tensor must be at least matrix!");
            #endif
        }
        Tensor<U> result(left._size);
        for(int i = 0; i < left._size; i++) {
            Tensor tmp = matmul(left[i], right[i]);
            result[i]._rank = tmp._rank;
            result[i]._size = tmp._size;
            result[i] = std::move(tmp);
        }
        result._rank = result[0]._rank + 1;
        return result;
    }

    // -- else _rank == 2 --

    int M = left._size;                      // rows in left mat
    int K = left[0]._size;           // cols in left mat
    int K2 = right._size;               // rows in right mat
    int N  = right[0]._size;   // cols in right mat

    #if DEBUG_TENSOR
        if (K != K2) {
            throw std::runtime_error("MatMul Error: Inner dimensions do not match (A(MxK) * B(KxN)).");
        }
    #endif

    // (M x N).
    Tensor<U> result(M, N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            U sum = static_cast<U>(0);
            for (int k = 0; k < K; k++) {
                sum += left[i][k].value() * right[k][j].value();
            }
            result[i][j] = sum;
        }
    }

    return result;
}
