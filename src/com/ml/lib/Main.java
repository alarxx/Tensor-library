package com.ml.lib;

import com.ml.lib.tensor.Tensor;

import static com.ml.lib.tensor.Tensor.tensor;

public class Main {
    public static void main(String[] args) {
        Tensor lr = tensor(0.1d);

        Tensor I = tensor(new double[][]{{2}}).requires_grad(true);

        Tensor K = tensor(new double[][]{{-5}});
        Tensor B = tensor(new double[][]{{0.5}});

        Tensor T;

        System.out.println("first_try: " + K.dot(I).add(B));
        System.out.println("Target: " + function(I));

        // Fit, LearningS
        for(int i=0; i<10000; i++){
            I = I.rand();
            T = function(I);

            Tensor Y = K.dot(I).add(B);

            Tensor Loss = T.sub(Y).pow(2);

            Loss._backward_();

            Tensor dK = lr.mul(K.getGrad());
            K = K.sub(dK);

            Tensor dB = lr.mul(B.getGrad());
            B = B.sub(dB);
        }

        I.get(0, 0).setScalar(2);
        System.out.println("last_try: " + K.dot(I).add(B));
        System.out.println("Target: " + function(I));

        System.out.println("K:"+K);
        System.out.println("B:"+B);
    }


    // f(x) = kx + b
    public static Tensor function(Tensor X){
        float   k = 3,
                b = 5;

        return X
                .mul(tensor(k))
                .add(tensor(b));
    }
}