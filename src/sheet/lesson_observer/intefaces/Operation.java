package sheet.lesson_observer.intefaces;

import com.ml.lib.tensor.Tensor;

public interface Operation {
    Tensor operation(Tensor tensor);
}
