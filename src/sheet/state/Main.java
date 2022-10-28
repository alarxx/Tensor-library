package sheet.state;

import com.ml.lib.Core;

public class Main {
    public static void main(String[] args) {
        TensorWState tensorWS = new TensorWState(Core.tensor(new float[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        }));

//        tensorWS.add(new TensorWState(Core.tensorWS(new float[][]{
//                {1, 1, 1},
//                {1, 1, 1},
//                {1, 1, 1}
//        })));
//        System.out.println(tensorWS.getState().getParent());
//        System.out.println(tensorWS.get());

        tensorWS.mirror();
        System.out.println(tensorWS.get());
    }
}
