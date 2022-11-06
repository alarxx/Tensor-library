package com.ml.lib.linear_algebra.operations;

import com.ml.lib.core.Core;
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
                {1, 2, 0, 2, 1},
                {1, 2, 0, 2, 1},
                {1, 2, 0, 2, 1},
                {1, 2, 0, 2, 1},
                {1, 2, 0, 2, 1},
        });
        tensor = tensor.div(tensor(2));

        Tensor kernel = tensor(new float[][]{
                {-1, 1},
                {-1, 1}
        });

        Tensor conv = new Conv(1, Type.AVG).apply(tensor, kernel);
        System.out.println("conv:"+conv);
    }

    public enum Type {
        SUM, AVG
    }

    protected final int step;
    protected final Type type;

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
        return new int[]{2, 2, 2};
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
    protected Tensor operation(Tensor matrix, Tensor kernel) {
        // Мы уверены, что matrix values для изображения in [0, 1]
        // Но если бы не были, то сделали так
//        float max = new MaxOfRank(2).apply(matrix).getScalar();
//        matrix = matrix.div(tensor(max));

        float[] numOfMinusAndAbs = type == Type.AVG ? numOfMinusAndAbs(kernel) : new float[]{1, 1};
        float m = numOfMinusAndAbs[0];
        float a = numOfMinusAndAbs[1];

        int[] resDims = getResultDims();
        int l = resDims.length;

        int     res_rows = resDims[l - 2],
                res_cols = resDims[l - 1];

        Tensor res = new Tensor(res_rows, res_cols);

        int     rows = matrix.dims()[0],
                cols = matrix.dims()[1],

                k_rows = kernel.dims()[0],
                k_cols = kernel.dims()[1];

        float n = k_rows * k_cols;

        for (int r = 0; r < res_rows; r++) {
            for (int c = 0; c < res_cols; c++) {

                float sum = 0;
                for (int i = 0; i < k_rows; i++) {
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

                if (type == Type.AVG) {
                    sum += m;
                    sum /= a;
                }

                res.set(tensor(sum), r, c);
            }
        }

        return res;
    }


    private float[] numOfMinusAndAbs(Tensor matrix){
        int     rows = matrix.dims()[0],
                cols = matrix.dims()[1];

        float minus = 0,
                all = 0;
        for(int r=0; r<rows; r++){
            for(int c=0; c<cols; c++){
                float value = matrix.get(r, c).getScalar();
                if(value < 0)
                    minus += Math.abs(value);
                all += Math.abs(value);
            }
        }

        return new float[]{minus, all};
    }
}
