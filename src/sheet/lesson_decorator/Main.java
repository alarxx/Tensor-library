package sheet.lesson_decorator;

import com.ml.lib.tensor.Tensor;

import static com.ml.lib.tensor.Tensor.tensor;

public class Main {
    public static void main(String[] args) {
        Tensor scalar = tensor(1f);

        // Последовательное применение разных скалярных операций.
        Operation op = new Plus(2, new Clone()); // 1+2=3
        op = new Mult(2, op); // 3*2=6
        op = new Divide(2, op); // 6/2=3
        op = new Minus(4, op); // 3-4=-1

        Tensor res = op.operation(scalar);

        System.out.println(res);
    }
}
