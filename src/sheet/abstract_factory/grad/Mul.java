package sheet.abstract_factory.methods;

import com.ml.lib.autograd.Method;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.Core.mul;

public class Mul implements Method {
    @Override
    public Tensor _forward_(Tensor tensor, Tensor other) {
        return mul(tensor, other);
    }

    @Override
    public void _backward_(Tensor grad, Tensor[] depends_on) {
        depends_on[0].getAutoGrad()._backward_(mul(depends_on[1], grad));
        depends_on[1].getAutoGrad()._backward_(mul(depends_on[0], grad));
    }
}
