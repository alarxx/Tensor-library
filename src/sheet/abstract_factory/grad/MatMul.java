package sheet.abstract_factory.grad;

import com.ml.lib.autograd.Method;
import com.ml.lib.interfaces.Operation;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.Core.dot;
import static com.ml.lib.Core.tr;

public class MatMul implements Operation {
    @Override
    public Tensor apply(Tensor src1, Tensor src2) {
        return src1.getAutoGrad()._method_(new com.ml.lib.autograd.methods.MatMul(), src2);
    }

    @Override
    public Tensor apply(Tensor src) {
        return src.getAutoGrad()._method_(new com.ml.lib.autograd.methods.MatMul(), null);
    }
}
