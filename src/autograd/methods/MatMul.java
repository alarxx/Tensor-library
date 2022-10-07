package lib.autograd.methods;

import lib.tensor.Tensor;
import lib.autograd.Method;

import static lib.Core.dot;
import static lib.Core.tr;

public class MatMul implements Method {
    @Override
    public Tensor _forward_(Tensor tensor, Tensor other) {
        return dot(tensor, other);
    }

    @Override
    public void _backward_(Tensor grad, Tensor[] depends_on) {
        depends_on[0].getAutoGrad()._backward_(dot(grad, tr(depends_on[1])));
        depends_on[1].getAutoGrad()._backward_(tr(dot(tr(grad), depends_on[0])));
    }
}
