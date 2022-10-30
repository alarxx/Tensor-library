package com.ml.lib.linear_algebra.operations.self_operation;

import com.ml.lib.Main;
import com.ml.lib.core.Core;
import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.tensor.Tensor;

import java.util.List;

import static com.ml.lib.tensor.Tensor.tensor;

public class MaxMin extends Operation {
    public static void main(String[] args) {
        MaxMin op = new MaxMin(2);
        Tensor tensor = tensor(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

        Tensor min = op.apply(tensor);
        System.out.println(min);
    }
    private int rank;
    public MaxMin(int rank){
        this.rank = rank;
    }
    @Override
    protected int[] ranksToCorrelate(Tensor src1, Tensor nll) {
        return new int[]{rank};
    }

    @Override
    protected int[] resultTensorsDims(Tensor src1, Tensor nll) {
        int[] dims = src1.dims().clone();

        for(int i=0; i<rank; i++)
            dims[dims.length - 1 + i] = 1;

        return dims;
    }

    @Override
    protected Tensor operation(Tensor tensor, Tensor nll) {
        int     rows = tensor.dims()[0],
                cols = tensor.dims()[1];

        float max = Float.MIN_VALUE;
        float min = Float.MAX_VALUE;

        List<Tensor> lt = Core.allTensorOfRank(tensor, rank);
        for(Tensor t: lt){
            float[] maxmin = new float[]{max, min};
            maxmin(tensor, maxmin);
        }

        return null;
    }

    private void maxmin(Tensor tensor, float[] maxmin){
        if(tensor.isScalar()){
            float val = tensor.getScalar();
            float max = maxmin[0];
            float min = maxmin[1];
            if(val < min){
                maxmin[1] = val;
            }
            if(val > max){
                maxmin[0] = val;
            }
        }
        else{
            for(int i=0; i<tensor.getLength(); i++){
                maxmin(tensor.get(i), maxmin);
            }
        }
    }
}
