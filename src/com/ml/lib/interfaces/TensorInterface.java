package com.ml.lib.interfaces;

import com.ml.lib.tensor.Tensor;

public interface TensorInterface {
    int[] dims();

    // Интерфейс зависимый от реализации... это бесполезный интерфейс...
    Tensor set(Tensor item, int ... indexes);

    Tensor get(int ... indexes);

    Tensor fill(double item);

    int getLength();

    boolean isScalar();

    double getScalar();
    Tensor setScalar(double scalar);

    int rank();

    Tensor clone();
}
