package lib.linear_algebra.operations;


import lib.linear_algebra.Operation;
import lib.tensor.Tensor;

public class Transposition extends Operation {
    @Override
    protected int[] ranksToCorrelate(Tensor src1, Tensor src2) {
        return new int[]{2};
    }

    @Override
    protected int[] resultTensorsDims(Tensor src1, Tensor src2) {
        int[] dims = src1.dims().clone();

        int buf = dims[0];
        dims[0] = dims[1];
        dims[1] = buf;

        return dims;
    }

    @Override
    protected Tensor operation(Tensor matrix, Tensor nll) {
        int     rows = matrix.getLength(),
                cols = matrix.get(0).getLength();

        Tensor transposed = new Tensor(cols, rows);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                Tensor t = matrix.get(r, c);
                transposed.set(t, c, r);
            }
        }

        return transposed;
    }

}



