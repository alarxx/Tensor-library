package lib.autograd.methods;

import lib.tensor.Tensor;
import lib.autograd.Method;

public class Conv implements Method {
    @Override
    public Tensor _forward_(Tensor tensor, Tensor other) {
        return null;
    }

    @Override
    public void _backward_(Tensor grad, Tensor[] depends_on) {

    }
}
