package sheet.generic_try;

import sheet.generic_try.linear_algebra.Tensor;
import sheet.generic_try.linear_algebra.functions.Sum;

public class Main {
    public static void main(String[] args) {
        Tensor<Float> t1 = new Tensor<>(2, 2);
        Tensor<Float> t2 = new Tensor<>(2, 2);
        for(int i=0; i<2; i++) {
            for (int j = 0; j < 2; j++) {
                t1.set((float) i * j, i, j);
                t2.set((float)i*j, i, j);
            }
        }
        Sum sum = new Sum();
        Tensor t3 = sum.function(t1, t2);
        System.out.println(t3.get(0, 0));
    }
}