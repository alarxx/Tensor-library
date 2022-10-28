package com.ml.lib.linear_algebra.operations.self_operation;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.tensor.Tensor;


/**
 * [[
 *      [1, 2],
 *      [3, 4]
 * ],
 * [
 *      [5, 6],
 *      [7, 8]
 * ]]
 * 2d+ compression
 * => [
 *      [6, 8],
 *      [10, 12]
 *    ]
 *
 *
 *    IDK how to write it yet
 * */
public class Compression extends Operation {
    @Override
    protected int[] ranksToCorrelate(Tensor src1, Tensor src2) {
        return new int[]{0}; // idk yet
    }

    @Override
    protected int[] resultTensorsDims(Tensor src1, Tensor src2) {
        return new int[0];
    }

    @Override
    protected Tensor operation(Tensor t1, Tensor t2) {
        return null;
    }
}
