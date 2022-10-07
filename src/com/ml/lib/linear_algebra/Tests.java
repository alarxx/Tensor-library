package com.ml.lib.linear_algebra;

import com.ml.lib.Core;
import com.ml.lib.linear_algebra.operations.MatMul;
import com.ml.lib.linear_algebra.operations.Transposition;
import com.ml.lib.linear_algebra.operations.elementary.Sum;
import com.ml.lib.tensor.Tensor;

import java.util.Arrays;

public class Tests {

    public static void main(String[] args) {
        transposeTestMat1x2();
    }

    private static void tensorFactory(){
        float[][][][] d4d = new float[][][][]{
                {
                        {
                                {1, 2},
                                {3, 4}
                        },
                        {
                                {5, 6},
                                {7, 8}
                        }
                },
                {
                        {
                                {9, 10},
                                {11, 12}
                        },
                        {
                                {13, 14},
                                {15, 16}
                        }
                }
        };

        Tensor imageTensor = Core.tensor(d4d);

        System.out.println(Arrays.toString(imageTensor.dims()));
        System.out.println(imageTensor);
    }


    private static void castToDimsTest(){
        int     depth = 3,
                rows = 2,
                cols = 3;

        Tensor tensor = new Tensor(depth, rows, cols);

        for(int d=0, v=0; d<depth; d++){
            for(int r=0; r<rows; r++){
                for(int c=0; c<cols; c++, v++){
                    tensor.set(Core.tensor(v), d, r, c);
                }
            }
        }
        System.out.println(tensor);

        Tensor t = Core.castToDims(tensor, 2, 3, 3);
        System.out.println(t);
    }


    public static void sumFunctionTest(){
        float[]     vector = new float[]{1, 2};

        float[][]   mat1 = new float[][]{{1, 2}, {3, 4}},

                    mat2 = new float[][]{{5, 6}, {7, 8}},

                    colVector = new float[][]{{1}, {2}};

        float[][][] d3d = new float[][][]{ {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}} };

        Tensor  mat_t1 = Core.tensor(mat1),
                mat_t2 = Core.tensor(mat2),
                vector_t = Core.tensor(vector),
                col_vec_t3 = Core.tensor(colVector),
                d3d_t = Core.tensor(d3d);

        System.out.println("mat1:" + mat_t1);
        System.out.println("mat2:" + mat_t2);
        System.out.println("vector:" + vector_t);
        System.out.println("col_vec:" + col_vec_t3);
        System.out.println("3d:" + d3d_t + "\n");


        Sum sum = new Sum();

        Tensor res;

        res = sum.apply(mat_t1, vector_t);
        System.out.println("mat1 + vector:" + res + "\n");

        res = sum.apply(mat_t1, mat_t2);
        System.out.println("mat1 + mat2:" + res + "\n");

        res = sum.apply(col_vec_t3, col_vec_t3);
        System.out.println("colVec+colVec:" + res + "\n");

        res = sum.apply(mat_t1, Core.tensor(1.1f));
        System.out.println("mat1 + 1.1f:" + res + "\n");

        res = sum.apply(Core.tensor(1.1f), mat_t1);
        System.out.println("1.1f + mat1:" + res + "\n");


        res = sum.apply(d3d_t, mat_t1);
        System.out.println("3d+mat1:" + res);

    }


    private static void matMulMatTest(){
        MatMul matMul = new MatMul();

        float[][] mat1 = new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        };
        Tensor matT1 = Core.tensor(mat1);

        float[][] mat2 = new float[][]{
                {1, 2},
                {3, 4},
                {5, 6}
        };
        Tensor matT2 = Core.tensor(mat2);

        Tensor result = matMul.apply(matT1, matT2);

        System.out.println(matT1);
        System.out.println(matT2);
        System.out.println(result);
    }
    private static void matMul3DTest(){

        MatMul matMul = new MatMul();

        float[][][] mat1 = new float[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6}
                },
                {
                        {7, 8, 9},
                        {10, 11, 12}
                }
        };
        Tensor matT1 = Core.tensor(mat1);

        float[][][] mat2 = new float[][][]{
                {
                        {1, 2},
                        {3, 4},
                        {5, 6}
                },
                {
                        {7, 8},
                        {9, 10},
                        {11, 12}
                }
        };

        Tensor matT2 = Core.tensor(mat2);

        Tensor result = matMul.apply(matT1, matT2);

        System.out.println(matT1);
        System.out.println(matT2);
        System.out.println(result);

    }


    public static void upTheRankTest(){
        float[][] matArr = new float[][]{{1, 2}, {3, 4}};
        Tensor mat = Core.tensor(matArr);

        Tensor d3d = Core.upTheRank(mat, 3, true);

        d3d.set(Core.tensor(11f), 0, 0, 0);

        System.out.println(Arrays.toString(mat.dims()));
        System.out.println(mat);
        System.out.println(Arrays.toString(d3d.dims()));
        System.out.println(d3d);
    }


    private static void transposeTestMat2x2(){
        float[][] matArr = new float[][]{{1, 2}, {3, 4}};
        Tensor mat = Core.tensor(matArr);
        System.out.println(mat);

        Tensor mat_tr = new Transposition().apply(mat);

        System.out.println(mat_tr);
    }
    private static void transposeTestMat1x2(){
        float[][] matArr = new float[][]{{1}, {2}};
        Tensor mat = Core.tensor(matArr);
        System.out.println(mat);

        Tensor mat_tr = new Transposition().apply(mat);

        System.out.println(mat_tr);
    }
    private static void transposeTestVector(){
        float[] matArr = new float[]{1, 2};
        Tensor mat = Core.tensor(matArr);
        System.out.println(mat);

        Tensor mat_tr = new Transposition().apply(mat);

        System.out.println(mat_tr);
    }


}
