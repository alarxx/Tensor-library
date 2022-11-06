package com.ml.lib.autograd.methods;

import com.ml.lib.autograd.OperationGrad;
import com.ml.lib.tensor.Tensor;

/**
 * Not completed
 * */
public class Div implements OperationGrad {
    @Override
    public Tensor _forward_(Tensor tensor, Tensor other) {
        return null;
    }

    @Override
    public void _backward_(Tensor grad, Tensor[] depends_on) {

    }
}
