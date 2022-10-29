package com.ml.lib.autograd.methods;

import com.ml.lib.tensor.Tensor;
import com.ml.lib.autograd.OperationGrad;

import static com.ml.lib.Core.sum;

public class Sum implements OperationGrad {
    @Override
    public Tensor _forward_(Tensor tensor, Tensor other) {
        return sum(tensor, other, false);
    }

    @Override
    public void _backward_(Tensor grad, Tensor[] depends_on) {
        depends_on[0].getAutoGrad()._backward_(grad);
        depends_on[1].getAutoGrad()._backward_(grad);
    }
}
