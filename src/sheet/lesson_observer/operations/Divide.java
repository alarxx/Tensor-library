package sheet.lesson_observer.operations;

import com.ml.lib.tensor.Tensor;
import sheet.lesson_observer.intefaces.Operation;

import static com.ml.lib.Core.tensor;


public class Divide implements Operation {
    private float v;

    public Divide(float v){
        this.v = v;
    }

    @Override
    public Tensor operation(Tensor t) {
        return tensor(t.getScalar()/v);
    }
}
