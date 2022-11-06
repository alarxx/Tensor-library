package com.ml.lib.linear_algebra.operations.morph_transform;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.tensor.Tensor.tensor;

public class Erosion extends Operation {

    public static void main(String[] args) {
        Tensor tensor = tensor(new double[][]{{

        }});
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

                newRows = (rows - k_rows) + 1, //1 + ((mat - kernel) / step);
                newCols = (cols - k_cols) + 1;

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

                float res_val = 1;
                m:for (int i = 0; i < k_rows; i++) {
                    for (int j = 0; j < k_cols; j++) {
                        double value = matrix.get(r + i, c + j).getScalar();
                        if(value == 0){
                            res_val = 0;
                            break m;
                        }
                    }
                }

                res.set(tensor(res_val), r, c);
            }
        }

        return res;
    }
}
