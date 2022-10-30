package sheet.state;

import com.ml.lib.core.Core;

public class Transposed extends State {
    public Transposed(TensorWState parent){
        super(parent);
    }

    @Override
    public TensorWState get() {
        return new TensorWState(Core.tr(parent.getTensor()));
    }
}
