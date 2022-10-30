package sheet.state;

import com.ml.lib.core.Core;

import static com.ml.lib.tensor.Tensor.tensor;

public class Main {
    public static void main(String[] args) {
        TensorWState tensorWS = new TensorWState(tensor(new float[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        }));

        tensorWS.add(new TensorWState(tensor(new float[][]{
                {1, 1, 1},
                {1, 1, 1},
                {1, 1, 1}
        })));
        System.out.println(tensorWS.get());

        tensorWS.mirror();
        System.out.println(tensorWS.get());

        tensorWS.rotate();
        System.out.println(tensorWS.get());
    }
}
