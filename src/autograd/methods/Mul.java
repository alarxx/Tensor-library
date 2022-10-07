package lib.autograd.methods;

import lib.tensor.Tensor;
import lib.autograd.Method;

import static lib.Core.mul;

public class Mul implements Method {
    @Override
    public Tensor _forward_(Tensor tensor, Tensor other) {
        return mul(tensor, other);
    }

    @Override
    public void _backward_(Tensor grad, Tensor[] depends_on) {
        depends_on[0].getAutoGrad()._backward_(mul(depends_on[1], grad));
        depends_on[1].getAutoGrad()._backward_(mul(depends_on[0], grad));
    }
}
