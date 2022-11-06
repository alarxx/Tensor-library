package com.ml.lib.linear_algebra.operations.elementary;

import com.ml.lib.tensor.Tensor;

import static com.ml.lib.tensor.Tensor.tensor;

public class Test {
    public static void main(String[] args) {
        Tensor mat = tensor(new double[][][]{{
                {1, 2, 3, 11},
                {4, 5, 6, 12},
                {7, 8, 9, 13},
                {7, 8, 9, 14},
        }
        });

        Tensor colvec = tensor(new double[][]{
                {1},
                {2},
                {3},
                {4},
        });
//        Tensor rowvec = tensor(new float[][]{{1, 2, 3, 4}});

        Tensor result = new Sum().apply(mat, colvec);
        System.out.println(result);
    }
}
