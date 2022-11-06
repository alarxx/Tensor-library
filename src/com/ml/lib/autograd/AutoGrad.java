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
    public static Tensor div(Tensor t1, Tensor t2){
        return null;
    }
    public static Tensor dot(Tensor t1, Tensor t2){
        return t1.getAutoGrad()._method_(new MatMul(), t2);
    }

    public static Tensor conv(Tensor tensor, Tensor kernel){
        return  tensor.getAutoGrad()._method_(new Conv(), kernel);
    }

    public static Tensor neg(Tensor t1){
        return t1.getAutoGrad()._method_(new Neg());
    }
    public static Tensor pow(Tensor tensor, float pow){
        return tensor.getAutoGrad()._method_(new Pow(pow));
    }

    private Tensor tensor;
    private Tensor grad;

    private Tensor[] depends_on;

    private OperationGrad operationGrad; // Creation Operation


    //Если функция расчитывается второй раз, прошлый градиент нужно сбросить
    private boolean clean_grad;


    public AutoGrad(Tensor tensor){
        this.tensor = tensor;
    }


    private Tensor initGrad(OperationGrad operationGrad, Tensor[] depends_on) {
        this.operationGrad = operationGrad;

        this.depends_on = depends_on;

        grad = new Tensor(tensor.dims()).fill(0).requires_grad(false);

        return tensor;
    }


    // Например свертка, матричные умножения и тд
    public Tensor _method_(OperationGrad operationGrad, Tensor other){
        // Должны очистить градиент от прошлого распространения градиента.
        // Это происходит если была передача вперед, задействующая этот Тензор.
        // При распространении назад градиент складывается
        setCleanGrad(true);

        // Обычная операция
        Tensor result = operationGrad._forward_(tensor, other);

        // Результирующий тензор должен требовать градиент теперь
        result.requires_grad(true);

        // инициализируем результирующему тензору каким методом он был создан и родительские тензора, их количество в пределе [1, 2]
        result.getAutoGrad().initGrad(operationGrad, new Tensor[]{tensor, other});

        return result;
    }
    // Например, транспонирование
    public Tensor _method_(OperationGrad operationGrad){
        return this._method_(operationGrad, null);
    }

    /**
     * Не рекомендую использовать этот метод вне пакета
     * Из-за ./method_examples, пришлось сделать метод public
     * */
    public void _backward_(Tensor backward_grad) {
        if(clean_grad)
            clean_grad();

        grad = grad.add(backward_grad);

        if(operationGrad != null)
            operationGrad._backward_(grad, depends_on);
    }

    /** Первая производная всегда равна 1 */
    @Override
    public void _backward_(){
        setCleanGrad(true);
        _backward_(new Tensor(tensor.dims()).fill(1));
    }

    @Override
    public Tensor getGrad(){
        return grad;
    }

    private void setCleanGrad(boolean b){
        // надо ли назад тоже отдавать приказ очистки?
        clean_grad = b;
    }
    private boolean getCleanGrad(){
        return clean_grad;
    }
    private void clean_grad(){
        if(grad == null)
            initGrad(null, null);

        grad.fill(0);
        setCleanGrad(false);

        if(depends_on != null) {
            depends_on[0].getAutoGrad().clean_grad();
            if (depends_on.length > 1)
                depends_on[1].getAutoGrad().clean_grad();
        }
    }
}
