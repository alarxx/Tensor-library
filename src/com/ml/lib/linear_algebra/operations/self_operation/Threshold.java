package com.ml.lib.linear_algebra.operations.self_operation;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.core.Core.numberOfElements;
import static com.ml.lib.tensor.Tensor.tensor;

public class Threshold extends Operation {
    public static void main(String[] args) {
        Tensor tensor = tensor(new float[][]{
                {0.1f, 0.2f, 0.8f},
                {0.05f, 0.3f, 0.5f},
                {0.6f, 0.9f, 1f},
        });
//        Tensor tensor = tensor(new float[]{0.1f, 0.2f, 0.8f});

        tensor = new Threshold(0.5f).apply(tensor);
        System.out.println(tensor);
    }

    private float threshold;
    public Threshold(float threshold){
        this.threshold = threshold;
    }
    @Override
    protected int[] ranksToCorrelate(Tensor src1, Tensor nll) {
        int[] ranks = new int[]{0, -1, 0};

        // За счет того, что нам сразу передаются матрицы, мы избегаем прохождения по элементам этих матриц дважды.
        if(src1.rank() >= 2){
            ranks = new int[]{2, -1, 2};
        }

        return ranks;
    }


    @Override
    protected int[] resultTensorsDims(Tensor src, Tensor nll) {
        return src.dims().clone();
    }

    @Override
    public Tensor operation(Tensor tensor, Tensor nll) {
        if (getRanks()[0] == 0) { // Это скаляр
            return tensor(tensor.getScalar() > threshold ? 1 : 0);
        }
        else { // Это матрицы
            int     rows = tensor.getLength(),
                    cols = tensor.get(0).getLength();

            Tensor result = new Tensor(rows, cols);
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    Tensor scalarT = tensor.get(r, c);
                    Tensor resT = result.get(r, c);

                    if(scalarT.getScalar() > threshold){
                        resT.setScalar(1f);
                    }
                    else resT.setScalar(0f);
                }
            }
            return result;
        }
    }
}
