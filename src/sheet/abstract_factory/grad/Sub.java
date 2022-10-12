package sheet.abstract_factory.methods;

import com.ml.lib.autograd.Method;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.Core.neg;
import static com.ml.lib.Core.sub;

public class Sub implements Method {
    @Override
    public Tensor _forward_(Tensor tensor, Tensor other) {
        return sub(tensor, other);
    }

    @Override
    public void _backward_(Tensor grad, Tensor[] depends_on) {
        depends_on[0].getAutoGrad()._backward_(grad);
        depends_on[1].getAutoGrad()._backward_(neg(grad));
    }
}
