package com.ml.lib.linear_algebra.operations.self_operation;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.tensor.Tensor;

import java.util.Arrays;

import static com.ml.lib.tensor.Tensor.tensor;

public class MaxOfRank extends Operation {
    public static void main(String[] args) {
        Tensor tensor = tensor(new double[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6}
                },
                {
                        {1, 2, 3},
                        {4, 5, 6}
                },
        });

        Tensor max = new MaxOfRank(2).apply(tensor);
        System.out.println(Arrays.toString(max.dims()));
        System.out.println(max);
    }

    private int rank;
    private int[] subDims;
    public MaxOfRank(int rank){
        this.rank = rank;
    }

    @Override
    protected int[] ranksToCorrelate(Tensor src1, Tensor nll) {
        return new int[]{rank, -1, rank};
    }

    @Override
    protected int[] resultTensorsDims(Tensor src1, Tensor nll) {
        this.subDims = new int[rank];

        int[] dims = src1.dims().clone();
        for(int i=0; i<rank; i++){
            dims[dims.length - i - 1] = 1;
            subDims[i] = 1;
        }
//        System.out.println(Arrays.toString(subDims));
        return dims;
    }

    @Override
    protected Tensor operation(Tensor tensor, Tensor nll) {
        Tensor result = new Tensor(subDims);
        result.get(0, 0).setScalar(max(tensor, Double.MIN_VALUE));
        return result;
    }

    private double max(Tensor tensor, double maxVal){
        if(tensor.isScalar()){
            return Math.max(tensor.getScalar(), maxVal);
        }
        else{
            for(Tensor t: tensor){
                maxVal = max(t, maxVal);
            }
            return maxVal;
        }
    }
}
