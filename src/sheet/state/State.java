package sheet.state;

import com.ml.lib.tensor.Tensor;

public abstract class State {
    protected TensorWState parent, additional;


    public State(TensorWState parent){
        setParent(parent);
    }

    public State(TensorWState parent, TensorWState additional){
        setParent(parent);
        setAdditional(additional);
    }


    abstract public TensorWState get();




    public TensorWState getParent() {
        return parent;
    }
    public void setParent(TensorWState parent) {
        this.parent = parent;
    }

    public TensorWState getAdditional() {
        return additional;
    }
    public void setAdditional(TensorWState additional) {
        this.additional = additional;
    }
}
