package sheet.state;

import com.ml.lib.Core;
import com.ml.lib.tensor.Tensor;

public class Transposed extends State {
    public Transposed(Tensor parent){
        super(parent);
    }

    @Override
    public Tensor get() {
        return Core.tr(parent);
    }
}
