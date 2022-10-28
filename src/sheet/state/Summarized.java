package sheet.state;

import com.ml.lib.Core;
import com.ml.lib.tensor.Tensor;

public class Summarized extends State {
    public Summarized(Tensor parent, Tensor additional){
        super(parent, additional);
    }

    @Override
    public Tensor get() {
        return Core.sum(parent, additional);
    }
}