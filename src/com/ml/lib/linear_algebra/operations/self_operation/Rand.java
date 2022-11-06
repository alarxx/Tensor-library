package com.ml.lib.linear_algebra.operations.self_operation;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.tensor.Tensor;

import java.util.Random;

import static com.ml.lib.tensor.Tensor.tensor;

public class Rand extends Operation {
    public static void main(String[] args) {
        Tensor tensor = new Tensor(3, 3);

        tensor = new Rand(-5, 5).apply(tensor);
        System.out.println(tensor);
    }

    private final double min, max;
    public Rand(){
        this(0, 1);
    }
    public Rand(double min, double max){
        this.min = min;
        this.max = max;
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
            return tensor(Math.random() * (max - min) + min);
        }
        else { // Это матрицы
            int     rows = tensor.getLength(),
                    cols = tensor.get(0).getLength();

            Tensor result = new Tensor(rows, cols);

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    result.get(r, c).setScalar(Math.random() * (max - min) + min);
                }
            }
            return result;
        }
    }
}
