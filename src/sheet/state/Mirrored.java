package sheet.state;

import com.ml.lib.tensor.Tensor;
import com.ml.lib.linear_algebra.operations.self_operation.Mirror;

public class Mirrored extends State {
    public Mirrored(TensorWState parent){
        super(parent);
    }

    @Override
    public TensorWState get() {
        return new TensorWState(new Mirror().apply(parent.getTensor()));
    }
}