package sheet.lesson_decorator;

import com.ml.lib.tensor.Tensor;

public interface Operation {
    Tensor operation(Tensor tensor);
}
