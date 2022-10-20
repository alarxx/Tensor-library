package com.ml.lib.linear_algebra.operations;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.tensor.Tensor;

public class Conv extends Operation {
    @Override
    protected int[] ranksToCorrelate(Tensor tensor, Tensor kernels) {
        return new int[]{2, 2};
    }

    @Override
    protected int[] resultTensorsDims(Tensor tensor, Tensor kernels) {
        // Здесь нужно прямо расчитывать, потому что размерность после свертки матрицы уменьшается, в зависимости от ядра
        return tensor.dims().clone();
    }

    @Override
    protected Tensor operation(Tensor matrix, Tensor kernel) {
        return null;
    }
}
