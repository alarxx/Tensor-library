package lib;

import lib.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static lib.Core.*;

public class Main {
    public static void main(String[] args) {
        Tensor a = tensor(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

        Tensor b = tensor(new float[][]{
                {2, 3},
                {4, 5},
                {6, 7}
        });

        Tensor c = dot(a, b, true);
        System.out.println(c);

        c._backward_();

        System.out.println("c_der:"  + c.getGrad());
        System.out.println("a_der:"  + a.getGrad());
        System.out.println("b_der:"  + b.getGrad());
    }
}