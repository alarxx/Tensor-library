package com.ml.lib.linear_algebra.operations.self_operation;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.tensor.Tensor;

import java.util.Arrays;

import static com.ml.lib.Core.tensor;

public class Mirror extends Operation {
    public static void main(String[] args) {
        Tensor tensor = tensor(new float[][]{
                {0, 1, 2, 3},
                {0, 4, 5, 6},
                {0, 7, 8, 9}
        });

        Mirror mirror = new Mirror();
        Tensor rotated = mirror.apply(tensor);

        System.out.println(rotated);
    }

    //---------SINGLETON------------------
    private static Operation instance;
    public static Operation getInstance(){
        if(instance == null){
            instance = new Transposition();
        }
        return instance;
    }
    //-------------------------------------

    @Override
    protected int[] ranksToCorrelate(Tensor src1, Tensor src2) {
        return new int[]{2};
    }

    @Override
    protected int[] resultTensorsDims(Tensor src1, Tensor src2) {
        return src1.dims().clone();
    }

    @Override
    protected Tensor operation(Tensor matrix, Tensor nll) {
        int     rows = matrix.getLength(),
                cols = matrix.get(0).getLength();

        Tensor mirror = new Tensor(rows, cols);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                Tensor t = matrix.get(r, cols - c - 1);
                mirror.set(t, r, c);
            }
        }

        return mirror;
    }


}
