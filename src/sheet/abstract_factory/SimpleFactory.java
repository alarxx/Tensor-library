package sheet.abstract_factory;


public class OperationFactory {

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
