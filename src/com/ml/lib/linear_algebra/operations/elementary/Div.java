package com.ml.lib.linear_algebra.operations.elementary;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.linear_algebra.operations.MatMul;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.tensor.Tensor.tensor;

public class Div extends ElementByElement {
    public static void main(String[] args) {
        Tensor tensor = tensor(new double[][][]{{
                {1, 2, 3, 4, 5},
                {1, 2, 3, 4, 5},
                {1, 2, 3, 4, 5},
                {1, 2, 3, 4, 5},
                {1, 2, 3, 4, 5},
        }});

        Tensor d = tensor(new double[][]{
                {2d}
        });

        Tensor result = new Div().apply(tensor, d);
        System.out.println("result:"+result);
    }
    //---------SINGLETON------------------
    private static Operation instance;
    public static Operation getInstance(){
        if(instance == null){
            instance = new Div();
        }
        return instance;
    }
    //-------------------------------------

    @Override
    public double operation(double a, double b){
        return a / b;
    }
}