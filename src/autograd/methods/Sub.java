package lib.autograd.methods;

import lib.tensor.Tensor;
import lib.autograd.Method;

import static lib.Core.neg;
import static lib.Core.sub;

public class Sub implements Method {
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
