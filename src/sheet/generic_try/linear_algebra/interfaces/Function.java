package sheet.generic_try.linear_algebra.interfaces;

import sheet.generic_try.linear_algebra.Tensor;

public interface Function {
    Tensor function(Tensor t1, Tensor t2);
}
