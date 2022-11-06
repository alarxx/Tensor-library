package com.ml.lib.nn;

import com.ml.lib.tensor.Tensor;

import static com.ml.lib.tensor.Tensor.tensor;

public class Main {
    public static void main(String[] args) {
        Tensor lr = tensor(0.1d);

        Tensor I = tensor(new double[][]{
                {2},
        }).requires_grad(true);

        Tensor W = tensor(new double[][]{
                {0.5},
        });
        Tensor B = tensor(new double[][]{
                {0.5}
        });

        Tensor T = function(I);

        System.out.println(W.dot(I).add(B));
        System.out.println(T);


        for(int i=0; i<2000; i++){
            I = I.rand();
            T = function(I);

            Tensor Y = W.dot(I).add(B);

            Tensor L = T.sub(Y).pow(2);

            L._backward_();

            Tensor dW = lr.mul(W.getGrad());
            W = W.sub(dW);

            Tensor dB = lr.mul(B.getGrad());
            B = B.sub(dB);
        }

        I.get(0, 0).setScalar(2);
        System.out.println(W.dot(I).add(B));
        System.out.println(function(I));

        System.out.println("W:"+W);
        System.out.println("B:"+B);
    }


    // f(I) = 3*I + 5
    public static Tensor function(Tensor X){
        return X
                .mul(tensor(3))
                .add(tensor(5));
    }
}
