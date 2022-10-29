package com.ml.lib.linear_algebra.operations;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.tensor.Tensor;

import java.util.Arrays;

import static com.ml.lib.Core.tensor;

/**
 * Not completed
 * */
public class Conv extends Operation {
    public static void main(String[] args) {
        Tensor tensor = tensor(new float[][]{
                {1, 2, 3, 4, 5},
                {1, 2, 3, 4, 5},
                {1, 2, 3, 4, 5},
                {1, 2, 3, 4, 5},
                {1, 2, 3, 4, 5},
        });

        Tensor kernel = tensor(new float[][]{
                {1, 1, 1},
                {1, 1, 1},
                {1, 1, 1}
        });

        Tensor conv = new Conv(1, Type.NONE).apply(tensor, kernel);

        System.out.println("conv:"+conv);
    }

    public enum Type {
        NONE, AVG, RATE
    }

    private final int step;
    private final Type type;

    public Conv(int step, Type type){
        this.step = step;
        this.type = type;
    }
    public Conv(int step){
        this(step, Type.NONE);
    }
    public Conv(Type type){
        this(1, type);
    }
    public Conv(){
        this(1, Type.NONE);
    }

    @Override
    protected int[] ranksToCorrelate(Tensor tensor, Tensor kernel) {
        return new int[]{2, 2};
    }

    @Override
    protected int[] resultTensorsDims(Tensor tensor, Tensor kernel) {
        // Здесь нужно прямо расчитывать, потому что размерность после свертки матрицы уменьшается, в зависимости от ядра
        int[] dims = tensor.dims().clone();
        int l = dims.length;

        int     rows = dims[l-2],
                cols = dims[l-1],

                k_rows = kernel.dims()[kernel.dims().length - 2],
                k_cols = kernel.dims()[kernel.dims().length - 1],

                newRows = rows - k_rows + 1,
                newCols = cols - k_cols + 1;

        dims[l-2] = newRows;// row
        dims[l-1] = newCols;// col

        return dims;
    }


    @Override
    protected Tensor operation(Tensor matrix, Tensor kernel) {
//        if(type){}
        int[]   resDims = getResultDims();
        int     l = resDims.length;

        int     res_rows = resDims[l-2],
                res_cols = resDims[l-1];

        Tensor res = new Tensor(res_rows, res_cols);

        int     rows = matrix.dims()[0],
                cols = matrix.dims()[1],

                k_rows = kernel.dims()[0],
                k_cols = kernel.dims()[1];

        float n = k_rows * k_cols;

        for(int r=0; r<res_rows; r++){
            for(int c=0; c<res_cols; c++){

                float sum = 0;
                for(int i=0; i<k_rows; i++) {
                    for (int j = 0; j < k_cols; j++) {
                        sum += kernel
                                .get(i, j).getScalar()
                                *
                                matrix
                                        .get(
                                            r + r * (step - 1) + i,
                                            c + c * (step - 1) + j
                                        ).getScalar();
                    }
                }

                res.set(tensor(sum/(n)), r, c);
            }
        }

        return res;
    }
}
