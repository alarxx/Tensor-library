package com.ml.lib.autograd.methods;

import com.ml.lib.autograd.OperationGrad;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.core.Core.*;
import static com.ml.lib.tensor.Tensor.tensor;

/**
 * Not completed
 * */
public class Pow implements OperationGrad {
    public static void main(String[] args) {
        Tensor tensor = tensor(new float[]{2, 3});
        Tensor Pow = tensor.getAutoGrad()._method_(new Pow(2));
        Pow._backward_();
        System.out.println(Pow.getGrad());
    }
    private float pow;
    public Pow(float pow){
        this.pow = pow;
    }

    @Override
    public Tensor _forward_(Tensor tensor, Tensor nll) {
        return pow(tensor, pow);
    }

    @Override
    public void _backward_(Tensor grad, Tensor[] depends_on) {
        depends_on[0].getAutoGrad()._backward_(grad.mul( depends_on[0].pow(pow - 1).mul(tensor(pow)) ));
//        depends_on[1].getAutoGrad()._backward_(neg(grad));
    }
}