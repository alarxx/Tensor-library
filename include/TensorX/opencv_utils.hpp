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
#ifndef _TENSOROPENCVUTILS_
#define _TENSORIMAGEIO_

#include <string>

#include <opencv2/opencv.hpp>

#include "TensorX/Tensor.hpp"


namespace tensor {

    // Converter
    template<Arithmetic T = float>
    Tensor<T> mat2tensor(const cv::Mat& src){
        if (src.empty()){
            throw std::runtime_error("mat2tensor: empty Mat");
        }
        // CV_8UC1 - Grayscale
        if(src.type() == CV_8UC1){
            int rows = src.rows;
            int cols = src.cols;

            Tensor<T> t(rows, cols);

            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    t.get(r, c) = src.at<uchar>(r, c);
                }
            }

            return t;
        }
        // CV_8UC3, CV_8UC4 etc.
        else {
            throw std::runtime_error("mat2tensor: unsupported image format");
        }
    }

    // Converter
    template<Arithmetic T = float>
    cv::Mat tensor2mat(const Tensor<T>& t){
        // 2D Tensor - Grayscale
        if (t.getRank() == 2){
            int rows = t.getDims()[0];
            int cols = t.getDims()[1];

            cv::Mat m(rows, cols, CV_8UC1);
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    m.at<uchar>(r, c) = t.get(r, c);
                }
            }
            return m;
        }
        // 3D Tensor
        else {
            throw std::runtime_error("tensor2mat: unsupported image format");
        }
    }

    // Read Image
    template<Arithmetic T = float>
    Tensor<T> imread_gray(std::string path){
        cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE); // IMREAD_COLOR (BGR), IMREAD_UNCHANGED (BGRA)
        // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        // std::cout << image.size() << "x" << image.channels() << std::endl;

        if (!image.data){
            throw std::runtime_error("read_gray: No image data");
        }

        // Convert OpenCV Mat to Tensor
        Tensor t = mat2tensor(image);
        // std::cout << "t: rank(" << t.getRank() << "), size(" << t.getLength() << ")" << std::endl;
        // std::cout << "t: dims(" << array2string(t.getRank(), t.getDims()) << ")" << std::endl;
        // std::cout << "t: type(" << typeid(typename decltype(t)::type).name() << ")" << std::endl;
        return t;
    }

    // Visualize
    template<Arithmetic T>
    void imshow(Tensor<T> t, std::string title = "Display Image"){
        cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
        // Convert Tensor to OpenCV Mat
        cv::imshow(title, tensor2mat(t));
        cv::waitKey(0);
    }

};

#endif



