package com.ml.lib.autograd.methods;

import com.ml.lib.tensor.Tensor;
import com.ml.lib.autograd.OperationGrad;

import java.util.Arrays;

import static com.ml.lib.core.Core.neg;
import static com.ml.lib.core.Core.sub;

public class Sub implements OperationGrad {
    @Override
    public Tensor _forward_(Tensor tensor, Tensor other) {
        return sub(tensor, other);
    }

    @Override
    public void _backward_(Tensor grad, Tensor[] depends_on) {
        depends_on[0].getAutoGrad()._backward_(grad);
        depends_on[1].getAutoGrad()._backward_(neg(grad));
    }
}
