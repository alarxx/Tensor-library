package com.ml.lib.linear_algebra.operations.elementary;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.linear_algebra.operations.MatMul;

public class Sub extends ElementByElement {
    //---------SINGLETON------------------
    private static Operation instance;
    public static Operation getInstance(){
        if(instance == null){
            instance = new Sub();
        }
        return instance;
    }
    //-------------------------------------

    @Override
    public float operation(float a, float b){
        return a - b;
    }

}