package sheet.lesson_observer.operations;

import com.ml.lib.tensor.Tensor;
import sheet.lesson_observer.intefaces.Operation;

import static com.ml.lib.tensor.Tensor.tensor;

public class Sub implements Operation {
    private float v;

    public Sub(float v) {
        this.v = v;
    }

    @Override
    public Tensor operation(Tensor t) {
        double value = t.getScalar() - v;
//        return new Tensor().setScalar(value);
        return tensor(value);
    }
}
