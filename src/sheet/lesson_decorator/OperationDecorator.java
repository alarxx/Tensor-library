package sheet.lesson_decorator;


import com.ml.lib.tensor.Tensor;

public class OperationDecorator implements Operation {
    private Operation operation;

    public OperationDecorator(Operation operation){
        this.operation = operation;
    }

    @Override
    public Tensor operation(Tensor tensorOfRank) {
        Tensor result = operation.operation(tensorOfRank);
        System.out.println(result);
        return result;
    }

}
