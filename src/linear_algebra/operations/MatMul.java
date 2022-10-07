package lib.linear_algebra.operations;

import lib.linear_algebra.Operation;
import lib.tensor.Tensor;

import static lib.Core.*;

public class MatMul extends Operation {

    public static void main(String[] args) {
        Tensor t1 = new Tensor(3, 2, 2).fill(1);
        Tensor t2 = new Tensor(2, 4).fill(2);

        System.out.println("t1:" + t1);
        System.out.println("t2:" + t2);

        MatMul matMul = new MatMul();

        Tensor result = matMul.apply(t1, t2);

        System.out.println("result: " + result);
    }

    @Override
    protected int[] ranksToCorrelate(Tensor src1, Tensor src2) {
        return new int[]{2, 2};
    }

    @Override
    protected int[] resultTensorsDims(Tensor src1, Tensor src2) {
        Tensor  mat1 = firstTensorOfRank(src1, 2),
                mat2 = firstTensorOfRank(src2, 2);

        int[] resDims;

        if(src1.rank() > src2.rank()){
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
