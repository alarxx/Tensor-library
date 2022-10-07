package com.ml.lib.autograd;

import com.ml.lib.Core;
import com.ml.lib.tensor.Tensor;

public class Tests {
    public static void main(String[] args) {

        Tensor a = Core.tensor(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

        Tensor b = Core.tensor(new float[][]{
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

}
