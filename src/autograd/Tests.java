package lib.autograd;

import lib.tensor.Tensor;

import static lib.Core.tensor;
import static lib.autograd.AutoGrad.*;
public class Tests {
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

        Tensor c = dot(a, b);
        System.out.println(c);

        c._backward_();

        System.out.println("c_der:"  + c.getGrad());
        System.out.println("a_der:"  + a.getGrad());
        System.out.println("b_der:"  + b.getGrad());
    }

}
