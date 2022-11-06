package com.ml.lib.autograd;

import com.ml.lib.autograd.methods.MatMul;
import com.ml.lib.core.Core;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.tensor.Tensor.tensor;

public class Tests {
    public static void main(String[] args) {
    }

    public static void shortened(){
        Tensor a = tensor(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

        Tensor b = tensor(new float[][]{
                {2, 3},
                {4, 5},
                {6, 7}
        });

        Tensor c = AutoGrad.dot(a, b);
        System.out.println(c);

        c._backward_();

        System.out.println("c_der:"  + c.getGrad());
        System.out.println("a_der:"  + a.getGrad());
        System.out.println("b_der:"  + b.getGrad());
    }

    public static void full(){
        Tensor a = tensor(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

        Tensor b = tensor(new float[][]{
                {2, 3},
                {4, 5},
                {6, 7}
        });

        Tensor c = a.getAutoGrad()._method_(new MatMul(), b);
        System.out.println(c);

        c._backward_();

        System.out.println("c_der:"  + c.getGrad());
        System.out.println("a_der:"  + a.getGrad());
        System.out.println("b_der:"  + b.getGrad());
    }

}
