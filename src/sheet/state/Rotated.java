package sheet.state;

import com.ml.lib.tensor.Tensor;
import com.ml.lib.linear_algebra.operations.self_operation.Rotate;

public class Rotated extends State {
    public Rotated(Tensor parent){
        super(parent);
    }

    @Override
    public Tensor get() {
        return new Rotate().apply(parent);
    }
}