package sheet.state;

import com.ml.lib.Core;

public class Main {
    public static void main(String[] args) {
        TensorWState tensor = new TensorWState(3, 3);

        tensor.set(Core.tensor(new float[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        }));

        tensor.sum(Core.tensor(new float[][]{
                {1, 1, 1},
                {1, 1, 1},
                {1, 1, 1}
        }));

        System.out.println(tensor.get());
    }
}
