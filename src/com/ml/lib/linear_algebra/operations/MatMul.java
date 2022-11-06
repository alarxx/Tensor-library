package com.ml.lib.linear_algebra.operations;

import com.ml.lib.core.Core;
import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.core.Core.*;
import static com.ml.lib.tensor.Tensor.tensor;

public class MatMul extends Operation {
    public static void main(String[] args) {
        Tensor mats1 = tensor(new float[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6}
                },
                {
                        {7, 8, 9},
                        {10, 11, 12}
                },
        });

        Tensor mats2 = tensor(new float[][][]{
                {
                        {1, 2},
                        {3, 4},
                        {5, 6}
                },
        });

        Tensor result = new MatMul().apply(mats1, mats2);

        System.out.println("mat1:"+mats1);
        System.out.println("mat2:"+mats2);
        System.out.println("result:"+result);
    }

    //---------SINGLETON------------------
    private static Operation instance;
    public static Operation getInstance(){
        if(instance == null){
            instance = new MatMul();
        }
        return instance;
    }
    //-------------------------------------

    @Override
    protected int[] ranksToCorrelate(Tensor src1, Tensor src2) {
        return new int[]{2, 2, 2};
    }

    @Override
    protected int[] resultTensorsDims(Tensor src1, Tensor src2) {
        Tensor  mat1 = firstTensorOfRank(src1, 2),
                mat2 = firstTensorOfRank(src2, 2);

        int[] resDims;

        // Берем самое большое количество матриц
        if(numberOfElements(src1.dims(), 2) > numberOfElements(src2.dims(), 2)){
            resDims = src1.dims().clone();
        } else {
            resDims = src2.dims().clone();
        }

        int     row1 = mat1.getLength(),
                col1 = mat1.get(0).getLength(),
                row2 = mat2.getLength(),
                col2 = mat2.get(0).getLength();

        if(col1 != row2){
            throwError("Matrices do not match");
        }

        resDims[resDims.length - 2] = row1;
        resDims[resDims.length - 1] = col2;

        return resDims;
    }

    @Override
    protected Tensor operation(Tensor mat1, Tensor mat2) {
        int     row1 = mat1.getLength(),
                col1 = mat1.get(0).getLength(),
//                row2 = mat2.getLength(),
                col2 = mat2.get(0).getLength();

        Tensor result = new Tensor(row1, col2);

        for(int r=0; r<row1; r++){
            for(int c=0; c<col2; c++){

                float sum = 0;
                for(int s=0; s<col1; s++){
                    float v = mat1.get(r, s).getScalar() * mat2.get(s, c).getScalar();
                    sum += v;
                }

                result.set(tensor(sum), r, c);
            }
        }

        return result;
    }
}
