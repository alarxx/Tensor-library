package com.ml.lib;

import com.ml.lib.tensor.Tensor;

import static com.ml.lib.Core.*;

public class Main {
    public static void main(String[] args) {
        Tensor a = tensor(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        }).requires_grad(true);

        Tensor b = tensor(new float[][]{
                {2, 3},
                {4, 5},
                {6, 7}
        });

        Tensor c = a.dot(b);
        System.out.println("c:"+c);

        c._backward_();

        System.out.println("c_der:"  + c.getGrad());
        System.out.println("a_der:"  + a.getGrad());
        System.out.println("b_der:"  + b.getGrad());
    }
}