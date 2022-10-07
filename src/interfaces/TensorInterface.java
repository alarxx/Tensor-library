package lib.interfaces;

import lib.tensor.Tensor;

public interface TensorInterface {
    int[] dims();

    // Интерфейс зависимый от реализации... это бесполезный интерфейс...
    Tensor set(Tensor item, int ... indexes);

    Tensor get(int ... indexes);

    Tensor fill(float item);

    int getLength();

    boolean isScalar();

    float getScalar();
    Tensor setScalar(float scalar);

    int rank();

    Tensor clone();
}
