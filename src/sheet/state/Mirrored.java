package sheet.state;

import com.ml.lib.tensor.Tensor;
import com.ml.lib.linear_algebra.operations.self_operation.Mirror;

public class Mirrored extends State {
    public Mirrored(Tensor parent){
        super(parent);
    }

    @Override
    public Tensor get() {
        return new Mirror().apply(parent);
    }
}