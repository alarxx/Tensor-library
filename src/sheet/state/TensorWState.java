package sheet.state;

import com.ml.lib.tensor.Tensor;

/**
 * Хотелось бы добавить такой вид записи:
 * Tensor tensor = ...;
 * tensor.rotate();
 * tensor.tr();
 * tensor.add(another1);
 * tensor.mul(another2);
 * tensor.dot(another3);
 *
 * Но не позволяет метод set and get,
 * мы не можем просто взять и координально изменить Tensor.
 *
 * Поэтому можно создать обертку Tensor-а и добавить какой-то state,
 * который бы давал последнее изменение этого тензора.
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
        state = new Rotated(get());

        return this;
    }
    public TensorWState mirror() {
        state = new Mirrored(get());

        return this;
    }
    public TensorWState add(TensorWState another){
        state = new Summarized(get(), another);

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
