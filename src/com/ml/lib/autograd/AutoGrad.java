package com.ml.lib.autograd;

import com.ml.lib.autograd.methods.*;
import com.ml.lib.tensor.Tensor;
import com.ml.lib.interfaces.AutoGradInterface;

public class AutoGrad implements AutoGradInterface {
    public static Tensor sum(Tensor t1, Tensor t2){
        return t1.getAutoGrad()._method_(new Sum(), t2);
    }
    public static Tensor sub(Tensor t1, Tensor t2){
        return t1.getAutoGrad()._method_(new Sub(), t2);
    }
    public static Tensor mul(Tensor t1, Tensor t2){
        return t1.getAutoGrad()._method_(new Mul(), t2);
    }

    public static Tensor dot(Tensor t1, Tensor t2){
        return t1.getAutoGrad()._method_(new MatMul(), t2);
    }

    public static Tensor neg(Tensor t1){
        return t1.getAutoGrad()._method_(new Neg());
    }

    private Tensor tensor;
    private Tensor grad;

    private Tensor[] depends_on;

    private Method method;


    //Если функция расчитывается второй раз, прошлый градиент нужно сбросить
    private boolean clean_grad;


    public AutoGrad(Tensor tensor){
        this.tensor = tensor;
    }


    private Tensor initGrad(Method method, Tensor[] depends_on) {
        this.method = method;

        this.depends_on = depends_on;

        grad = new Tensor(tensor.dims()).fill(0);

        return tensor;
    }


    // Например свертка
    public Tensor _method_(Method method, Tensor other){
        setCleanGrad(true);

        return method
                ._forward_(tensor, other)
                .getAutoGrad()
                .initGrad(method, new Tensor[]{tensor, other});
    }
    public Tensor _method_(Method method){
        return this._method_(method, null);
    }

    /**
     * Не рекомендую использовать этот метод вне пакета
     * Из-за ./method_examples, пришлось сделать метод public
     * */
    public void _backward_(Tensor backward_grad) {
        if(getCleanGrad()) {
            clean_grad();
        }

        grad = sum(grad, backward_grad);

        if(method != null)
            method._backward_(grad, depends_on);
    }

    /** Первая производная всегда равна 1 */
    @Override
    public void _backward_(){
        setCleanGrad(true);
        _backward_(new Tensor(tensor.dims()).fill(1));
    }


    private void setCleanGrad(boolean b){
        // надо ли назад тоже отдавать приказ очистки?
        clean_grad = b;
    }
    private boolean getCleanGrad(){
        return clean_grad;
    }
    private void clean_grad(){
        if(grad == null){
            initGrad(null, null);
        }
        grad.fill(0);
        setCleanGrad(false);

        if(depends_on != null) {
            depends_on[0].getAutoGrad().setCleanGrad(true);
            if (depends_on.length > 1)
                depends_on[1].getAutoGrad().setCleanGrad(true);
        }
    }

    @Override
    public Tensor getGrad(){
        return grad;
    }
}
