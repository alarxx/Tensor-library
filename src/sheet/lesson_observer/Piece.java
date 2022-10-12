package sheet.lesson_observer;

import com.ml.lib.tensor.Tensor;

import sheet.lesson_observer.intefaces.Observer;
import sheet.lesson_observer.intefaces.Operation;

public class Piece implements Observer {
    private Tensor tensor;

    public Piece(Tensor tensor){
        this.tensor = tensor;
    }

    public Tensor getTensor(){
        return tensor;
    }

    @Override
    public Tensor apply(Operation operation) {
        tensor = operation.operation(tensor);
        System.out.println(tensor);
        return tensor;
    }
}
