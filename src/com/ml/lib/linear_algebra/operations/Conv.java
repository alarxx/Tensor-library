package com.ml.lib.linear_algebra.operations;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.core.Core.conv;
import static com.ml.lib.tensor.Tensor.tensor;

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

        Tensor conv = new Conv(1, Type.SUM).apply(tensor, kernel);

        System.out.println("conv:"+conv);
    }

    public enum Type {
        SUM, AVG, RATE
    }

    private final int step;
    private final Type type;

    public Conv(int step, Type type){
        this.step = step;
        this.type = type;
    }
    public Conv(int step){
        this(step, Type.SUM);
    }
    public Conv(Type type){
        this(1, type);
    }
    public Conv(){
        this(1, Type.SUM);
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

                newRows = (rows - k_rows) / step + 1, //1 + ((mat - kernel) / step);
                newCols = (cols - k_cols) / step + 1;

        dims[l-2] = newRows;// row
        dims[l-1] = newCols;// col

        return dims;
    }

    @Override
    protected Tensor[] preconversion(Tensor tensor, Tensor kernel){
        Tensor[] t_arr = super.preconversion(tensor, kernel);

        if(type == Type.RATE){
            Tensor sum = conv(kernel, new Tensor(kernel.dims()).fill(1), 1, Type.SUM, false);
//            System.out.println(Arrays.toString(sum.dims()));
            Tensor k1 = kernel.div(sum); // Если kernel был req_grad, то и k1 тоже будет.
            t_arr[1] = k1;
        }

        return t_arr;
    }

    @Override
    protected Tensor operation(Tensor matrix, Tensor kernel) {
        System.out.println("mat:" + matrix);
        System.out.println("kernel:" + kernel);

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
                                matrix.get(
                                            r + r * (step - 1) + i,
                                            c + c * (step - 1) + j
                                        ).getScalar();
                    }
                }

                float val = sum;

                if(type == Type.AVG){
                    val /= n;
                }

                res.set(tensor(val), r, c);
            }
        }

        return res;
    }
}
