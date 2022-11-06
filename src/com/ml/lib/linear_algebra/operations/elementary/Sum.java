package com.ml.lib.linear_algebra.operations.elementary;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.linear_algebra.operations.MatMul;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.tensor.Tensor.tensor;

public class Sum extends ElementByElement {
    //---------SINGLETON------------------
    private static Operation instance;
    public static Operation getInstance(){
        if(instance == null){
            instance = new Sum();
        }
        return instance;
    }
    //-------------------------------------

    @Override
    public float operation(float a, float b){
        return a + b;
    }

}
