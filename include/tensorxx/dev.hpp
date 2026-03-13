#pragma once

#include <opencv2/opencv.hpp>

#include "Tensor.hpp"
#include "ops.hpp"
#include "opencv_utils.hpp"


namespace dev {

void test_custom_canny(){
    auto start = std::chrono::high_resolution_clock::now();

    // code block
    tensor::Tensor<float> t = tensor::imread_gray("./images/lenna.png");
    tensor::Tensor edges = tensor::canny(t, 30.0f, 80.0f);
    // tensor::imshow(edges);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "custom_canny: " << duration_ns.count() << "ns" << std::endl;
}

void test_opencv_canny(){
    auto start = std::chrono::high_resolution_clock::now();

    // OpenCV Canny
    cv::Mat img = cv::imread("./images/lenna.png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) return;
    cv::Mat edges;
    cv::Canny(img, edges, 100, 200);
    // cv::imshow("Original", img);
    // cv::imshow("Canny OpenCV", edges);
    // cv::waitKey(0);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "opencv_canny: " << duration_ns.count() << "ns" << std::endl;
}


template<typename T>
void draw_edge_components(const tensor::Tensor<T>& edges){
    // 1) Tensor -> Mat
    cv::Mat img = tensor::tensor2mat(edges);
    if (img.empty()) return;

    // 2) привести к 8-bit бинарной маске
    cv::Mat binary;

    if (img.type() != CV_8U) {
        img.convertTo(img, CV_32F);
        cv::Mat mask = (img > 0.0f);
        mask.convertTo(binary, CV_8U, 255.0);
    } else {
        cv::threshold(img, binary, 0, 255, cv::THRESH_BINARY);
    }

    // 3) найти контуры
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 4) сделать BGR картинку чтобы рисовать цветом
    cv::Mat canvas;
    cv::cvtColor(binary, canvas, cv::COLOR_GRAY2BGR);

    // 5) обвести каждую компоненту
    for (const auto& c : contours) {
        cv::Rect box = cv::boundingRect(c);
        cv::rectangle(canvas, box, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Edge Components", canvas);
    cv::waitKey(0);
}


void test(){
    tensor::test_conv();
    tensor::test_mul();
    tensor::test_mul_scalars();
    tensor::test_scalar_mul();
    // test_custom_canny();
    // test_opencv_canny();
}

};
