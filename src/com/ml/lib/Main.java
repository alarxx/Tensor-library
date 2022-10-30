package com.ml.lib;

import com.ml.lib.tensor.Tensor;

import java.time.Year;

import static com.ml.lib.tensor.Tensor.tensor;

public class Main {
    public static void main(String[] args) {
        Tensor s3 = tensor(3)
                .requires_grad(true);

        Tensor s7 = tensor(7)
                .requires_grad(true);

        Tensor s5 = tensor(5)
                .requires_grad(true);

        Tensor sc1 = s3.mul(s7);
        Tensor sc2 = s7.mul(s5);

        Tensor Y = sc1.mul(tensor(1f));
        Y._backward_();

        Y = sc1.add(sc2);
        Y._backward_();

        System.out.println("Y: " + Y.getGrad()); // 1
        System.out.println("sc1: " + sc1.getGrad()); // 1
        System.out.println("sc2: " + sc2.getGrad()); // 1
        System.out.println("s3: " + s3.getGrad()); // 7
        System.out.println("s7: " + s7.getGrad()); // 8
        System.out.println("s5: " + s5.getGrad()); // 7
    }
}