package com.ml.lib.linear_algebra.operations;

import com.ml.lib.core.Core;
import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.linear_algebra.operations.self_operation.MatMax;
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

        Tensor conv = new Conv(1, Type.RATE).apply(tensor, kernel);

        System.out.println("conv:"+conv);
    }

    public enum Type {
        SUM, AVG, K_SUM_1, RATE
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

        if(type == Type.K_SUM_1){
            Tensor sum = conv(kernel, new Tensor(kernel.dims()).fill(1), 1, Type.SUM, false);
//            System.out.println(Arrays.toString(sum.dims()));
            Tensor k1 = kernel.div(sum); // Если kernel был req_grad, то и k1 тоже будет.
            t_arr[1] = k1;
        }

        return t_arr;
    }

    @Override
    protected Tensor operation(Tensor matrix, Tensor kernel) {
//        System.out.println("mat:" + matrix);
//        System.out.println("kernel:" + kernel);

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

        if(type == Type.RATE) {
            float   orig_max = Core.max(matrix).getScalar(),
                    orig_min = Core.min(matrix).getScalar(),
                    orig_abs = Math.abs(orig_max - orig_min);

            float   res_max = Core.max(res).getScalar(),
                    res_min = Core.min(res).getScalar(),
                    res_abs = Math.abs(res_max - res_min);

            float orig_addi = orig_abs - orig_max; // 29 - 20 = 9
            float res_addi = res_abs - res_max; // 2 - 7 = -5

            for(int r=0; r<res_rows; r++) {
                for (int c = 0; c < res_cols; c++) {
                    float before = res.get(r, c).getScalar();
                    float after = ((before + res_addi) / res_abs) * orig_abs + orig_min;
                    res.set(tensor(after), r, c);
                }
            }
        }

        return res;
    }
}
