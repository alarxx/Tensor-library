package sheet.abstract_factory.grad;

import com.ml.lib.autograd.OperationGrad;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.Core.neg;
import static com.ml.lib.Core.sub;

public class Sub implements OperationGrad {
    @Override
    public Tensor _forward_(Tensor tensor, Tensor other) {
        return sub(tensor, other, false);
    }

    @Override
    public void _backward_(Tensor grad, Tensor[] depends_on) {
        depends_on[0].getAutoGrad()._backward_(grad);
        depends_on[1].getAutoGrad()._backward_(neg(grad, false));
    }
}
