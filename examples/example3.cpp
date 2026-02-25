#include <iostream>
#include <chrono>
#include <string>

#include <opencv2/opencv.hpp>

#include "TensorX/Tensor.hpp"

#include "TensorX/opencv_utils.hpp" // imread_gray, imshow
#include "TensorX/utils.hpp" // array2string
#include "TensorX/convolution.hpp" // conv
#include "TensorX/ops.hpp" // find_median, find_mean
#include "TensorX/image_processing.hpp"
#include "TensorX/dev.hpp" // test, draw_edge_components


using tensor::Tensor;


int main(){
    std::cout << "Hello, OpenCV!" << std::endl;
    dev::test();

    std::string path = std::string(IMAGE_DIR) + "/lenna.png";
    Tensor<float> t = tensor::imread_gray(path);

    std::cout << "t: rank(" << t.getRank() << "), size(" << t.getLength() << ")" << std::endl;
    std::cout << "t: dims(" << tensor::array2string(t.getRank(), t.getDims()) << ")" << std::endl;
    std::cout << "t: type(" << typeid(typename decltype(t)::type).name() << ")" << std::endl;

    // Convolution

    // Canny

    // 1. Gaussian Filter
    Tensor blurred = tensor::gaussian_blur(t, 1);

    // 2. Image Derivarive
    Tensor sobel = tensor::sobel_operator(blurred);

    // 3. Non-Maximum Suppression (NMS)
    Tensor nms = tensor::non_max_suppression(sobel);

    // 4. Double Thresholding
    Tensor strongweak = tensor::double_threshold(nms, 20.0f, 80.0f); // only for visualization here

    // 5. Hysterisis
    Tensor chained = tensor::hysterisis(nms, 20.0f, 80.0f);

    // Auto Threshold Values
    float median = find_median(sobel);
    float mean = find_mean(sobel);
    std::cout << "median: " << median << std::endl;
    std::cout << "mean: " << mean << std::endl;
    // float low = std::max(0.0f, mean);
    // float high = std::min(255.0f, std::min(80.0f, 2*mean));
    // Tensor strongweak = tensor::double_threshold(nms, low, high);
    // Tensor chained = tensor::hysterisis(nms, low, high);

    tensor::imshow(t);
    tensor::imshow(blurred, "Gaussian Blur 5x5");
    tensor::imshow(sobel, "Sobel Operator");
    tensor::imshow(nms, "Non-Maximum Suppression");
    tensor::imshow(strongweak, "Double Threshold");
    tensor::imshow(chained, "Hysterisis");
    // tensor::imshow(tensor::canny(t, 20.0f, 80.0f));

    dev::draw_edge_components(chained);

    return 0;
}


