package com.ml.lib.linear_algebra;

import com.ml.lib.core.Core;
import com.ml.lib.linear_algebra.operations.Conv;
import com.ml.lib.linear_algebra.operations.MatMul;
import com.ml.lib.linear_algebra.operations.self_operation.Transposition;
import com.ml.lib.linear_algebra.operations.elementary.Sum;
import com.ml.lib.tensor.Tensor;

import java.util.Arrays;

import static com.ml.lib.core.Core.*;
import static com.ml.lib.tensor.Tensor.tensor;

public class Tests {

    public static void main(String[] args) {
        matmulTest();
    }

    private static void rateKernel(){
        Tensor kernel = tensor(new float[][][]{
                {
                    {1, 2, 1},
                    {2, 4, 2},
                    {1, 2, 1}
                }
        });

        Tensor sum = conv(kernel, new Tensor(kernel.dims()).fill(1), 1, Conv.Type.SUM, false);

        Tensor k1 = kernel.div(sum); // Если kernel был req_grad, то и k1 тоже будет.

        System.out.println(k1);
    }

    private static void convolutionTest(){
        Tensor tensor = tensor(new float[][][]{
                {
                        {1, 2, 3, 4, 5},
                        {1, 2, 3, 4, 5},
                        {1, 2, 3, 4, 5},
                        {1, 2, 3, 4, 5},
                        {1, 2, 3, 4, 5},
                },
                {
                        {1, 2, 3, 4, 5},
                        {1, 2, 3, 4, 5},
                        {1, 2, 3, 4, 5},
                        {1, 2, 3, 4, 5},
                        {1, 2, 3, 4, 5},
                }
        });

        Tensor kernel = tensor(new float[][][]{
                {
                    {1, 1, 1},
                    {1, 1, 1},
                    {1, 1, 1}
                },
                {
                    {1, 2, 1},
                    {2, 4, 2},
                    {1, 2, 1}
                },
        });

        Tensor convolution = tensor.conv(kernel, 2, Conv.Type.AVG);
        System.out.println(convolution);
    }

    private static void trTest(){
        Tensor mat = tensor(new float[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6}
                },
                {
                        {1, 2, 3},
                        {4, 5, 6}
                }
        });

        mat = tr(mat);

        System.out.println(mat);
    }
    private static void singletonTest1(){
//        Operation matmul = new MatMul();
        Operation matmul = MatMul.getInstance();


        Tensor mat1 = tensor(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

        Tensor mat2 = tensor(new float[][]{
                        {1, 2},
                        {3, 4},
                        {5, 6}
        });

        Tensor result = matmul.apply(mat1, mat2);
//        Tensor result = Core.dot(mat1, mat2);

        System.out.println("mat1:"+mat1);
        System.out.println("mat2:"+mat2);
        System.out.println("result:"+result);

    }

    private static void matmulTest(){
        Tensor mat1 = tensor(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

        Tensor mat2 = tensor(new float[][]{
                {1, 2},
                {3, 4},
                {5, 6}
        });

        Tensor result = Core.dot(mat1, mat2, false);

        System.out.println("mat1:"+mat1);
        System.out.println("mat2:"+mat2);
        System.out.println("result:"+result);

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

        Tensor imageTensor = tensor(d4d);

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
                    tensor.set(tensor(v), d, r, c);
                }
            }
        }
        System.out.println(tensor);

        Tensor t = Core.castToDims(tensor, 2, 3, 3);
        System.out.println(t);
    }


    public static void sumOperationTest(){
        float[]     vector = new float[]{1, 2};

        float[][]   mat1 = new float[][]{{1, 2}, {3, 4}},

                    mat2 = new float[][]{{5, 6}, {7, 8}},

                    colVector = new float[][]{{1}, {2}};

        float[][][] d3d = new float[][][]{ {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}} };

        Tensor  mat_t1 = tensor(mat1),
                mat_t2 = tensor(mat2),
                vector_t = tensor(vector),
                col_vec_t3 = tensor(colVector),
                d3d_t = tensor(d3d);

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

        res = sum.apply(mat_t1, tensor(1.1f));
        System.out.println("mat1 + 1.1f:" + res + "\n");

        res = sum.apply(tensor(1.1f), mat_t1);
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
        Tensor matT1 = tensor(mat1);

        float[][] mat2 = new float[][]{
                {1, 2},
                {3, 4},
                {5, 6}
        };
        Tensor matT2 = tensor(mat2);

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
        Tensor matT1 = tensor(mat1);

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

        Tensor matT2 = tensor(mat2);

        Tensor result = matMul.apply(matT1, matT2);

        System.out.println(matT1);
        System.out.println(matT2);
        System.out.println(result);

    }


    public static void upTheRankTest(){
        float[][] matArr = new float[][]{{1, 2}, {3, 4}};
        Tensor mat = tensor(matArr);

        Tensor d3d = Core.upTheRank(mat, 3, true);

        d3d.set(tensor(11f), 0, 0, 0);

        System.out.println(Arrays.toString(mat.dims()));
        System.out.println(mat);
        System.out.println(Arrays.toString(d3d.dims()));
        System.out.println(d3d);
    }


    private static void transposeTestMat2x2(){
        float[][] matArr = new float[][]{{1, 2}, {3, 4}};
        Tensor mat = tensor(matArr);
        System.out.println(mat);

        Tensor mat_tr = new Transposition().apply(mat);

        System.out.println(mat_tr);
    }
    private static void transposeTestMat1x2(){
        float[][] matArr = new float[][]{{1}, {2}};
        Tensor mat = tensor(matArr);
        System.out.println(mat);

        Tensor mat_tr = new Transposition().apply(mat);

        System.out.println(mat_tr);
    }
    private static void transposeTestVector(){
        float[] matArr = new float[]{1, 2};
        Tensor mat = tensor(matArr);
        System.out.println(mat);

        Tensor mat_tr = new Transposition().apply(mat);

        System.out.println(mat_tr);
    }


}
