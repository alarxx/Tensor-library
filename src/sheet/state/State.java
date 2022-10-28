package sheet.state;

import com.ml.lib.tensor.Tensor;

public abstract class State {
    protected Tensor parent, additional;


    public State(Tensor parent){
        setParent(parent);
    }

    public State(Tensor parent, Tensor additional){
        setParent(parent);
        setAdditional(additional);
    }


    abstract public Tensor get();


    public Tensor getParent() {
        return parent;
    }
    public void setParent(Tensor parent) {
        this.parent = parent;
    }


    public Tensor getAdditional() {
        return additional;
    }
    public void setAdditional(Tensor additional) {
        this.additional = additional;
    }
}
