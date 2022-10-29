package sheet.state;

import com.ml.lib.tensor.Tensor;

/**
 * I would like to add this type of record:
 * Tensor tensor = ...;
 * tensor.rotate();
 * tensor.tr();
 * tensor.add(another1);
 * tensor.mul(another2);
 * tensor.dot(another3);
 *
 * But does not allow tensor style,
 * in particular  set and get methods,
 * we can't just take and radically change Tensor.
 *
 * Therefore, you can create a Tensor wrapper and add some state,
 * which would give the last change of this tensor.
 * For example, during transposition, the dimension of matrices changes,
 * and during matrix multiplication, the dimension of matrices also changes.
 * How, then, can we assign these matrices to a tensor without changing it cardinally,
 * in particular the dimensions? We can wrap Tensor.
 *
 * Why won't I add this to the project? This will add a lot of logic and lines of code. KISS.
 * */
public class TensorWState {
    private Tensor tensor;
    private State state;

    public TensorWState(Tensor tensor){
        setTensor(tensor);
    }

    public TensorWState tr() {
        state = new Transposed(get());

        return this;
    }
    public TensorWState rotate() {
//        state = new Rotated(get());
        state = new Rotated(this);

        return this;
    }
    public TensorWState mirror() {
//        state = new Mirrored(get());
        state = new Mirrored(this);

        return this;
    }
    public TensorWState add(TensorWState another){
//        state = new Summarized(get(), another);
        state = new Summarized(this, another);

        return this;
    }


    public TensorWState get() {
        if(state!=null){
            return state.get();
        }
        else{
            return this;
        }
    }

    public Tensor getTensor() {
        return tensor;
    }
    public void setTensor(Tensor tensor) {
        this.tensor = tensor;
    }

    public State getState() {
        return state;
    }
    public void setState(State state) {
        this.state = state;
    }


    @Override
    public String toString(){
        return getTensor().toString();
    }
}
