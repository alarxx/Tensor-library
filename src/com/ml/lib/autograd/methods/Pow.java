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
        Tensor A = tensor(3f);
        Tensor B = tensor(6f);

        Tensor C = A.getAutoGrad()._method_(new Pow(2));
        Tensor Y = C.getAutoGrad()._method_(new Mul(), B);
        Y._backward_();
        System.out.println(A.getGrad());
    }
    private double pow;
    public Pow(double pow){
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