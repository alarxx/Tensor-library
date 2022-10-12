package sheet.abstract_factory;

import com.ml.lib.interfaces.Operation;
import sheet.operation_simple_factory.OperationTypes;

public interface AbstractFactory {
    Operation createOperation(OperationTypes type);
}