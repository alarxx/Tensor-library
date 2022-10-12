package sheet.abstract_factory;


import com.ml.lib.interfaces.Operation;
import com.ml.lib.linear_algebra.operations.MatMul;
import com.ml.lib.linear_algebra.operations.elementary.*;
import sheet.operation_simple_factory.OperationTypes;

import static sheet.operation_simple_factory.OperationTypes.*;

public class SimpleFactory {

    Operation createOperation(OperationTypes type){
        if(type == sum)
            return (Operation) new Sum();
        else if(type == sub)
            return (Operation) new Sub();
        else if(type == mul)
            return (Operation) new Mul();
        else if(type == div)
            return (Operation) new Div();
        else if(type == matmul)
            return (Operation) new MatMul();
        return null;
    }
}
