package com.ml.lib.interfaces;

import com.ml.lib.tensor.Tensor;

public interface Operation {
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
