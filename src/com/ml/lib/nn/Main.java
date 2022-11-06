package com.ml.lib.nn;

import com.ml.lib.tensor.Tensor;

import static com.ml.lib.tensor.Tensor.tensor;

public class Main {
    public static void main(String[] args) {
        Tensor I = new Tensor(2, 1).requires_grad(true);
        Tensor W = new Tensor(2, 2);
        Tensor T = function(I);


        Tensor Y = W.dot(I);
        Tensor L = T.sub(Y);
        L = L.mul(L);
        L._backward_();

        System.out.println(L.getGrad());
        System.out.println(Y.getGrad());
    }

    // Tensor tensor = tensor(3f);
    // System.out.println(function(tensor));
    public static Tensor function(Tensor X){
        return X
                .mul(tensor(3));
//                .add(tensor(5f));
    }
}
