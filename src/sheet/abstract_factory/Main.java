package sheet.abstract_factory;

import com.ml.lib.interfaces.Operation;
import com.ml.lib.tensor.Tensor;
import sheet.operation_simple_factory.OperationFactory;
import sheet.operation_simple_factory.OperationTypes;

import static com.ml.lib.Core.tensor;

public class Main {
    public static void main(String[] args) {
        OperationFactory factory = new OperationFactory();

        Operation mul = factory.createOperation(OperationTypes.mul);

        Tensor t1 = tensor(new float[]{1, 2, 3});
        Tensor t2 = tensor(new float[]{1, 2, 3});

        Tensor result = mul.apply(t1, t2);

        System.out.println(result);
    }
}
