package sheet.lesson_observer.operations;

import com.ml.lib.tensor.Tensor;

import sheet.lesson_observer.intefaces.Operation;

import static com.ml.lib.tensor.Tensor.tensor;


public class Mult implements Operation {
    private float v;

    public Mult(float v){
        this.v = v;
    }

    @Override
    public Tensor operation(Tensor t) {
        return tensor(t.getScalar()*v);
    }
}
