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
#ifndef _TENSORIMAGEPROCESSING_
#define _TENSORIMAGEPROCESSING_

#include <queue>
#include <cmath>
#include <climits>

#include "TensorX/Tensor.hpp"
#include "ops.hpp"


namespace tensor {

    template<Arithmetic T>
    Tensor<T> gaussian_blur(const Tensor<T>& image, int times = 1){
        Tensor<float> _blur5x5 = {
            {2,  4,  5,  4, 2},
            {4,  9, 12,  9, 4},
            {5, 12, 15, 12, 5},
            {4,  9, 12,  9, 4},
            {2,  4,  5,  4, 2}
        };
        Tensor blur5x5 = div(159.0f, _blur5x5);

        Tensor<T> result = image;

        for (int i = 0; i < times; ++i){
            std::cout << "gaussian_blur" << std::endl;
            result = conv(result, blur5x5);
        }

        return result;
    }


    template<Arithmetic T>
    Tensor<T> sobel_operator(const Tensor<T>& image){
        Tensor<float> sobel_x = {
            {-1,  0,  1},
            {-2,  0,  2},
            {-1,  0,  1}
        };
        Tensor gx = conv(image, sobel_x);

        Tensor<float> sobel_y = {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };
        Tensor gy = conv(image, sobel_y);

        // magnitude
        int H = gx.getDims()[0];
        int W = gx.getDims()[1];

        Tensor<float> mag(H, W);

        for (int r = 0; r < H; ++r){
            for (int c = 0; c < W; ++c){
                float gxv = gx.get(r, c); // [-4*255; 4*255]
                float gyv = gy.get(r, c);
                // float magv = std::abs(gxv) + std::abs(gyv); // L1. [0; 8*255]
                float magv = std::sqrt(gxv * gxv + gyv * gyv); // L2. [0; 4*255]]
                mag.get(r, c) = magv;
            }
        }

        // normalization
        float maxv = find_max(mag.getLength(), mag.getCoeffs());
        Tensor out = mul(255/maxv, mag);

        return out;
    }


    template<Arithmetic T>
    Tensor<T> non_max_suppression(const Tensor<T>& image){
        int H = image.getDims()[0];
        int W = image.getDims()[1];

        Tensor<T> out(H, W);
        fill(0, out);

        for(int r = 1; r < H - 1; ++r){ // Don't count image boundaries
            for(int c = 1; c < W - 1; ++c){

                T center = image.get(r, c);

                bool keep_h = (center >= image.get(r, c - 1) && center >= image.get(r, c + 1));
                bool keep_v = (center >= image.get(r - 1, c) && center >= image.get(r + 1, c));
                bool keep_d1 = (center >= image.get(r - 1, c - 1) && center >= image.get(r + 1, c + 1));
                bool keep_d2 = (center >= image.get(r - 1, c + 1) && center >= image.get(r + 1, c - 1)); // /

                if(keep_h || keep_v || keep_d1 || keep_d2){
                    out.get(r, c) = center;
                }
                else {
                    out.get(r, c) = (T) 0;
                }
            }
        }
        return out;
    }


    template<Arithmetic T>
    Tensor<T> double_threshold(
            const Tensor<T>& image,
            const T low, const T high,
            const T WEAK = (T) 50, const T STRONG = (T) 255){
        int H = image.getDims()[0];
        int W = image.getDims()[1];

        Tensor<T> out(H, W);
        fill(0, out);

        for(int r = 0; r < H; ++r){
            for(int c = 0; c < W; ++c){
                T v = image.get(r, c);

                if(v >= high){
                    out.get(r, c) = STRONG;
                }
                else if(v >= low){
                    out.get(r, c) = WEAK;
                }
                else {
                    out.get(r, c) = (T) 0;
                }
            }
        }
        return out;
    }


    template<Arithmetic T>
    Tensor<T> hysterisis(
        const Tensor<T>& image,
        const T low, const T high,
        const T WEAK = (T) 50, const T STRONG = (T) 255
    ){
        struct Pixel {
            int x;
            int y;
            Pixel(int _x, int _y): x(_x), y(_y){}
        };

        int H = image.getDims()[0];
        int W = image.getDims()[1];

        // Double Threshold
        Tensor<T> strongweak(H, W);
        fill(0, strongweak);

        std::queue<Pixel> q; // strong pixels

        for(int r = 0; r < H; ++r){
            for(int c = 0; c < W; ++c){
                T v = image.get(r, c);

                if(v >= high){
                    strongweak.get(r, c) = STRONG;
                    q.emplace(r, c);
                }
                else if(v >= low){
                    strongweak.get(r, c) = WEAK;
                }
                else {
                    strongweak.get(r, c) = (T) 0;
                }
            }
        }

        // Hysterisis gets: strongweak, q
        Tensor<T> out(H, W);
        fill(0, out);
        // BFS
        while(!q.empty()){
            Pixel p = q.front();
            q.pop();

            // recursive behavior
            out.get(p.x, p.y) = STRONG;

            // 3x3 - 8-neighbor relationship
            for(int dx = -1; dx <= 1; ++dx){
                for(int dy = -1; dy <= 1; ++dy){
                    if(dx == 0 && dy == 0) continue;

                    int nx = p.x + dx;
                    int ny = p.y + dy;

                    if(nx < 0 || nx >= H || ny < 0 || ny >= W) continue;

                    if(strongweak.get(nx, ny) == WEAK){
                        strongweak.get(nx, ny) = STRONG;
                        // recursive behavior
                        q.emplace(nx, ny);
                    }
                }
            }
        }

        return out;
    }


    template<Arithmetic T>
    Tensor<T> canny(
        const Tensor<T>& image,
        const T low, const T high,
        const T WEAK = (T) 50, const T STRONG = (T) 255
    ){
        // Canny
        // 1. Gaussian Filter
        // Tensor blurred = tensor::gaussian_blur(image, 2);

        // 2. Image Derivarive
        Tensor sobel = tensor::sobel_operator(image);

        // 3. Non-Maximum Suppression (NMS)
        Tensor nms = tensor::non_max_suppression(sobel);

        // 4-5. Double Thresholding and Hysterisis
        Tensor chained = tensor::hysterisis(nms, low, high);

        return chained;
    }

};


#endif
