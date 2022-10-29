package sheet.state;

import com.ml.lib.linear_algebra.operations.self_operation.Mirror;
import com.ml.lib.tensor.Tensor;
import com.ml.lib.linear_algebra.operations.self_operation.Rotate;

public class Rotated extends State {
    public Rotated(TensorWState parent){
        super(parent);
    }

    @Override
    public TensorWState get() {
        return new TensorWState(new Rotate(90).apply(parent.getTensor()));

    }
}