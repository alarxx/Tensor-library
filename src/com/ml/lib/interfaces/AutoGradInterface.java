package com.ml.lib.interfaces;

import com.ml.lib.tensor.Tensor;

/**
 * The user should only use these methods
 * */
public interface AutoGradInterface {
    /** Метод вызывается на последнем значении операции функции. */
    void _backward_();
    Tensor getGrad();
}
