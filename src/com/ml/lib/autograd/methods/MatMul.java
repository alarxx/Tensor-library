package com.ml.lib.autograd.methods;

import com.ml.lib.tensor.Tensor;
import com.ml.lib.autograd.OperationGrad;

import static com.ml.lib.Core.dot;
import static com.ml.lib.Core.tr;

public class MatMul implements OperationGrad {
    @Override
    public Tensor _forward_(Tensor tensor, Tensor other) {
        return dot(tensor, other, false);
    }

    @Override
    public void _backward_(Tensor grad, Tensor[] depends_on) {
        depends_on[0].getAutoGrad()._backward_(dot(grad, tr(depends_on[1]), false));
        depends_on[1].getAutoGrad()._backward_(tr(dot(tr(grad), depends_on[0], false)));
    }
}
