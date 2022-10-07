package lib.interfaces;

import lib.tensor.Tensor;

public interface OperationInterface {
    /**
     * Метод не должен изменять состояние аргумента.
     * @param src2 may be null
     * */
    Tensor apply(Tensor src1, Tensor src2);

    /**
     *  Метод не должен изменять состояние аргумента.
     * @param src not null
     * */
    Tensor apply(Tensor src);
}
