package sheet.state;

import com.ml.lib.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class TensorWState extends Tensor{
    private State state;

    public TensorWState(int ... dims){
        super(dims);
    }

    public TensorWState tr() {
        state = new Transposed(get());

        return this;
    }
    public TensorWState rotate() {
        state = new Rotated(get());

        return this;
    }
    public TensorWState mirror() {
        state = new Mirrored(get());

        return this;
    }
    public TensorWState sum(Tensor another){
        state = new Summarized(this, another);

        return this;
    }


    public Tensor get() {
        if(state!=null){
            return state.get();
        }
        else{
            return this;
        }
    }
}
