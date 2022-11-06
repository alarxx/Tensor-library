package com.ml.lib.linear_algebra.operations.self_operation;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.tensor.Tensor.tensor;

public class Threshold extends Operation {
    public static void main(String[] args) {
        Tensor tensor = tensor(new float[][]{
                {0.1f, 0.2f, 0.8f},
                {0.05f, 0.3f, 0.5f},
                {0.6f, 0.9f, 1f},
        });

        tensor = new Threshold(0.5f).apply(tensor);
        System.out.println(tensor);
    }

    private float threshold;
    public Threshold(float threshold){
        this.threshold = threshold;
    }
    @Override
    protected int[] ranksToCorrelate(Tensor src, Tensor nll) {
        return new int[]{2, -1, 2};
    }

    @Override
    protected int[] resultTensorsDims(Tensor src, Tensor nll) {
        return src.dims().clone();
    }

    @Override
    protected Tensor operation(Tensor matrix, Tensor nll) {
        int     rows = matrix.getLength(),
                cols = matrix.get(0).getLength();
        Tensor result = new Tensor(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                Tensor scalarT = matrix.get(r, c);
                Tensor resT = result.get(r, c);

                if(scalarT.getScalar() > threshold){
                    resT.setScalar(1f);
                }
                else resT.setScalar(0f);
            }
        }
        return result;
    }
}
