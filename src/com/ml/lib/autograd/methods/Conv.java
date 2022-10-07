package com.ml.lib.autograd.methods;

import com.ml.lib.autograd.Method;
import com.ml.lib.tensor.Tensor;

public class Conv implements Method {
    @Override
    public Tensor _forward_(Tensor tensor, Tensor other) {
        return null;
    }

    @Override
    public void _backward_(Tensor grad, Tensor[] depends_on) {

    }
}
