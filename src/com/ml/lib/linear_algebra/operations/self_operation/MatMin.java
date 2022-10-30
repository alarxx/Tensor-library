package com.ml.lib.linear_algebra.operations.self_operation;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.tensor.Tensor.tensor;

public class MatMin extends Operation {
    public static void main(String[] args) {
        MatMin op = new MatMin();
        Tensor tensor = tensor(new float[][]{
                {1, 2, 3}, {4, 5, 6}
        });

        Tensor max = op.apply(tensor);
        System.out.println(max);
    }
    @Override
    protected int[] ranksToCorrelate(Tensor src1, Tensor nll) {
        return new int[]{2};
    }

    @Override
    protected int[] resultTensorsDims(Tensor src1, Tensor nll) {
        int[] dims = src1.dims().clone();
        dims[dims.length - 2] = 1;
        dims[dims.length - 1] = 1;
        return dims;
    }

    @Override
    protected Tensor operation(Tensor mat, Tensor nll) {
        int rows = mat.dims()[0], cols = mat.dims()[1];
        float max = Float.MAX_VALUE;
        for(int r=0; r<rows; r++){
            for(int c=0; c<cols; c++){
                float val = mat.get(r, c).getScalar();
                if(val < max){
                    max = val;
                }
            }
        }
        return tensor(max);
    }
}
