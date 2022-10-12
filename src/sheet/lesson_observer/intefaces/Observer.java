package sheet.lesson_observer.intefaces;


import com.ml.lib.tensor.Tensor;

public interface Observer {
    Tensor apply(Operation operation);
}
