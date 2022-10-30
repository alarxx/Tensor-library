package sheet.abstract_factory.grad;

import com.ml.lib.autograd.OperationGrad;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.core.Core.mul;

public class Mul implements OperationGrad {
    @Override
    public Tensor _forward_(Tensor tensor, Tensor other) {
        return mul(tensor, other, false);
    }

    @Override
    public void _backward_(Tensor grad, Tensor[] depends_on) {
        depends_on[0].getAutoGrad()._backward_(mul(depends_on[1], grad, false));
        depends_on[1].getAutoGrad()._backward_(mul(depends_on[0], grad, false));
    }
}
