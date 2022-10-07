package com.ml.lib.autograd.methods;

import com.ml.lib.tensor.Tensor;
import com.ml.lib.autograd.Method;

import static com.ml.lib.Core.neg;

public class Neg implements Method {
    @Override
    public Tensor _forward_(Tensor tensor, Tensor other) {
        return neg(tensor);
    }

    @Override
    public void _backward_(Tensor grad, Tensor[] depends_on) {
        depends_on[0].getAutoGrad()._backward_(neg(grad));
    }
}
