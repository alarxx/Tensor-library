package sheet.lesson_decorator;


import com.ml.lib.tensor.Tensor;

class Clone implements Operation {
    @Override
    public Tensor operation(Tensor t) {
        return t.clone();
    }
}
