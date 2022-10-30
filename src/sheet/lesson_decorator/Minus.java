package sheet.lesson_decorator;

import com.ml.lib.tensor.Tensor;

import static com.ml.lib.tensor.Tensor.tensor;

public class Minus extends OperationDecorator {
    private static final int RANK = 0;

    private float v;

    public Minus(float v, Operation operation) {
        super(operation);
        this.v = v;
    }

    @Override
    public Tensor operation(Tensor tensor) {
        Tensor t = super.operation(tensor);
        return tensor(t.getScalar() - v);
    }
}
