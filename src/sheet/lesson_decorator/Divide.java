package sheet.lesson_decorator;

import com.ml.lib.tensor.Tensor;

import static com.ml.lib.Core.tensor;

public class Divide extends OperationDecorator {
    private static final int RANK = 0;

    private float v;

    public Divide(float v, Operation operation) {
        super(operation);
        this.v = v;
    }

    @Override
    public Tensor operation(Tensor tensor) {
        Tensor t = super.operation(tensor);
        return tensor(t.getScalar() / v);
    }
}
