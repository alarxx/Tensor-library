package sheet.abstract_factory;

import com.ml.lib.interfaces.Operation;
import sheet.abstract_factory.grad.*;
import sheet.operation_simple_factory.OperationTypes;

import static sheet.operation_simple_factory.OperationTypes.*;

public class GradFactory implements AbstractFactory {

    @Override
    public Operation createOperation(OperationTypes type) {
        if(type == sum)
            return (Operation) new Sum();
        else if(type == sub)
            return (Operation) new Sub();
        else if(type == mul)
            return (Operation) new Mul();
        else if(type == matmul)
            return (Operation) new MatMul();
        return null;
    }
}
