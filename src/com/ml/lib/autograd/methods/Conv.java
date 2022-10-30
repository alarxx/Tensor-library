package com.ml.lib.autograd.methods;

import com.ml.lib.core.Core;
import com.ml.lib.autograd.OperationGrad;
import com.ml.lib.tensor.Tensor;

/**
 * Not completed
 * */
public class Conv implements OperationGrad {
    @Override
    public Tensor _forward_(Tensor tensor, Tensor kernel) {
        return Core.conv(tensor, kernel, false);
    }

    @Override
    public void _backward_(Tensor grad, Tensor[] depends_on) {

    }
}
