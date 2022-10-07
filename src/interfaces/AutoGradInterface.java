package lib.interfaces;

import lib.tensor.Tensor;

/**
 * The user should only use these methods
 * */
public interface AutoGradInterface {
    void _backward_();
    Tensor getGrad();
}
