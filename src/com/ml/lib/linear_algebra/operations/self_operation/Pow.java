package com.ml.lib.linear_algebra.operations.self_operation;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.tensor.Tensor.tensor;

public class Pow extends Operation {
    public static void main(String[] args) {
        Tensor tensor = tensor(new double[][]{
                {1, 2, 3},{4, 5, 6},{7, 8, 9}
        });
        Tensor s = new Pow(1/2f).apply(tensor);
        System.out.println(s);
    }
    private double pow;

    public Pow(double pow){
        this.pow = pow;
    }

    @Override
    protected int[] ranksToCorrelate(Tensor src, Tensor nll) {
        return new int[]{0, -1, 0};
    }

    @Override
    protected int[] resultTensorsDims(Tensor src, Tensor nll) {
        return src.dims().clone();
    }

    @Override
    protected Tensor operation(Tensor scalar, Tensor nll) {
        return tensor(Math.pow(scalar.getScalar(), pow));
    }
}
