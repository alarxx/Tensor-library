package lib.autograd.methods;

import lib.tensor.Tensor;
import lib.autograd.Method;

import static lib.Core.neg;

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
