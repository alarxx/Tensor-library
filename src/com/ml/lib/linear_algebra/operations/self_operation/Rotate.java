package com.ml.lib.linear_algebra.operations.self_operation;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.Core.tensor;

public class Rotate extends Operation {
    public static void main(String[] args) {
        Tensor tensor = tensor(new float[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });

        Rotate rotate = new Rotate();
        Tensor rotated = rotate.apply(tensor);

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
        int[] dims = src1.dims().clone();
        int l = dims.length;

        int buf = dims[l-1];
        dims[l-1] = dims[l-2];
        dims[l-2] = buf;

        return dims;
    }

    @Override
    protected Tensor operation(Tensor matrix, Tensor nll) {
        int     rows = matrix.getLength(),
                cols = matrix.get(0).getLength();

        Tensor rotated = new Tensor(rows, cols);

        // 90
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                Tensor t = matrix.get(r, c);
                rotated.set(t, c, rows - r - 1);
            }
        }

        return rotated;
    }


}
