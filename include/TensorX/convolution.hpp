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
#ifndef _TENSORCONVOLUTION_
#define _TENSORCONVOLUTION_

#include "TensorX/Tensor.hpp"
#include "ops.hpp"

#include <cassert>


namespace tensor {

    template<Arithmetic T>
    Tensor<T> conv(const Tensor<T>& image, const Tensor<T>& kernel){
        if(image.getRank() != 2 || kernel.getRank() != 2){
            throw std::runtime_error("conv: must have rank 2");
        }

        // image size
        int iH = image.getDims()[0];
        int iW = image.getDims()[1];
        // kernel size
        int kH = kernel.getDims()[0];
        int kW = kernel.getDims()[1];
        // output size
        int oH = iH - kH + 1;
        int oW = iW - kW + 1;

        Tensor<T> out(oH, oW);

        for(int r = 0; r < oH; ++r){
            for(int c = 0; c < oW; ++c){

                T sum = (T) 0;
                for(int kr = 0; kr < kH; ++kr){
                    for(int kc = 0; kc < kW; ++kc){
                        sum += image.get(r + kr, c + kc) * kernel.get(kr, kc);
                    }
                }
                out.get(r, c) = sum;

            }
        }

        return out;
    }

    void test_conv(){
        Tensor<float> image = {
            {1, 2, 3, 2},
            {1, 2, 3, 2},
            {1, 2, 3, 2},
            {1, 2, 3, 2}
        };
        Tensor<float> sobel_x = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        Tensor<float> expected = {
            {8, 0},
            {8, 0}
        };

        Tensor out = conv(image, sobel_x);
        // std::cout << c << std::endl;
        assert(equal(out, expected) && "conv result is not correct");
    }

};

#endif
