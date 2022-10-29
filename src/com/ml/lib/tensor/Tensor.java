package com.ml.lib.tensor;

import com.ml.lib.Core;
import com.ml.lib.autograd.AutoGrad;
import com.ml.lib.interfaces.AutoGradInterface;
import com.ml.lib.interfaces.TensorInterface;

import java.util.Arrays;
import java.util.Iterator;

import static com.ml.lib.Core.throwError;


/*** Final variant, I hope
 *
 * Tensor - array of Tensors, except rank-0 Tensor (scalar)
 *
 * [Tensor, Tensor, Tensor, Tensor]
 *    |       |       |       |
 *    |       |     [...]   [...]
 *    |  [Tensor, Tensor, ...]
 * [Tensor, Tensor...]
 *
 * Я не хочу писать дженерики, пока.
 * Есть проблема с дженериками в функциях, например:
 * как понимать какого типа будет результирующая матрица;
 * и скорость с ними будет меньше... хотя я и не претендую на скорость.
 *
 * В этой реализации, как и в JS, все числа(скаляры) - float значения.
 *
 * Tensor - scalar, только если dims = [],
 * но [1], [1, 1], [1, 1, 1]... не будут являться скалярами.
 *
 * В оригинальном варианте объявление Tensor-а - сложная операция, при создании обходится каждый элемент.
 * Хотелось сделать так, чтобы если тензор под индексом не затрагивается, то по умолчанию он null.
 * Для этого надо было добавить проверку на null в get/set, fill, toString,
 * а методы getScalar/setScalar не нуждаются в такой проверке:
 *      new Tensor().getScalar()/setScalar(),
 *      либо в связке с get - tensor.get(0).getScalar()/setScalar()
 *
 *
 * Добавлен Autograd
 * */
public class Tensor implements TensorInterface, AutoGradInterface, Iterable<Tensor> {

    private static final float INIT_VALUE = 0f;
    private Tensor[] array;
    private final int[] dims;
    private int length;

    // minimal, fundamental rank-0 Tensor - is a scalar
    private float scalar; //by default is zero
    private boolean isScalar = false;

    // Нужен для создания вложенных тензоров
    // Без геттеров сеттеров, используется только внутри класса
    private int[] subDims;


    public Tensor(int ... dims){
        setAutoGrad(new AutoGrad(this));
        this.dims = dims;

        if(dims.length == 0) { // scalar
            setLength(0);
            setIsScalar(true);
            setScalar(INIT_VALUE);
        }
        else {
            setLength(dims[0]);

            this.subDims = Arrays.copyOfRange(dims, 1, dims.length);

            array = new Tensor[getLength()];

//            for (int i = 0; i < getLength(); i++) {
//                array[i] = new Tensor(subDims); // recursion
//            }

//            Оптимизирован с помощью
//            if(array[id] == null){
//                array[id] = new Tensor(subDims);;
//            }
        }
    }

    @Override
    public Tensor get(int ... indexes){
        if(indexes.length == 0)
            return this;

        return getT(0, indexes);
    }

    private Tensor getT(int dimCursor, int ... indexes){
        int id = indexes[dimCursor];

        if(array[id] == null){
            array[id] = new Tensor(subDims);;
        }

        Tensor t = array[id];

        if(dimCursor == indexes.length - 1)
            return t;
        else
            return t.getT(dimCursor + 1, indexes); // this array is not the same as in the previous method from the recursion stack
    }


    // Альтернативный вариант, но длинный и непонятный, в другом нужно понять только метод get
    @Override
    public Tensor set(Tensor item, int... indexes) {
        if(indexes.length == 0)
            changeFields(item);
        else setT(0, item, indexes); // Метод может сработать, а может и нет

        return this;
    }

    private void setT(int dimCursor, Tensor item, int... indexes) {
        int id = indexes[dimCursor];

        if(array[id] == null){
            array[id] = new Tensor(subDims);;
        }

        if (dimCursor == indexes.length - 1) {
            if(dimsEqual(array[id], item)){
                array[id] = item;
            }
            else{
//                System.out.println(Arrays.toString(array[dim].dims()));
//                System.out.println(Arrays.toString(item.dims()));
                throwError("Dimensions are not match");
            }
        }
        else {
            array[id].setT(dimCursor + 1, item, indexes);
        }
    }

    /**
     * Так делать плохо? Метод должен менять только values of scalars по идее
     * Нельзя просто взять и координально поменять структуру тензора.
     * Если это так надо, то просто создайте новый тензор.
     * */
    private void changeFields(Tensor tensor){
        if(!Core.dimsEqual(this, tensor)){
            throwError("Dims are not equal");
        }
        this.array = tensor.array;
//        this.length = tensor.length;
        this.scalar = tensor.scalar;
//        this.isScalar = tensor.isScalar;
//        this.subDims = tensor.subDims;
    }

    // Можно использовать как пример как пробежаться по всем элементам
    @Override
    public Tensor fill(float value) {
        if(isScalar()){
            setScalar(value);
        }
        else {
            for (int i = 0; i < getLength(); i++) {
                if(array[i] == null){
                    array[i] = new Tensor(subDims);
                }

                Tensor t = array[i];
                t.fill(value);
            }
        }

        return this; // Нужно для builder паттерна
    }


    /**
     * Do not change values manually!
     * Clone the result if you need so.
     * */
    @Override
    public int[] dims(){
        return dims;
    }

    @Override
    public int getLength(){
        return length;
    }
    protected void setLength(int length){
        this.length = length;
    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();

        sb.append("tensor(\n");
        toString(sb);

        return sb.toString().trim().concat("\n)");
    }
    //Похож на fill по реализации
    private void toString(StringBuilder sb){
        if(isScalar()){
            sb.append(String.format("%.1f", getScalar())).append(' ');
        }
        else {
            for (int i = 0; i < getLength(); i++) {
                // если это 3d, то сперва выводится первая матрица, потом вторая и тд.
                // если это матрица, то сперва выводится первая строка...
                // строка закончилась ->'\n', матрица закончилась ->'\n', 3rd dim закончилось ->'\n'
                if(array[i] == null){
                    array[i] = new Tensor(subDims);
                }

                Tensor t = array[i];
                t.toString(sb);
            }
            sb.append('\n');
        }
    }

    @Override
    public boolean isScalar(){
        return isScalar;
    }
    protected void setIsScalar(boolean isScalar){
        this.isScalar = isScalar;
    }


    @Override
    public float getScalar(){
        if(!isScalar())
            throwError("Tensor is not a scalar");

        return scalar;
    }

    @Override
    public Tensor setScalar(float scalar) {
        if (!isScalar())
            throwError("Tensor is not a scalar");

        this.scalar = scalar;

        return this;
    }

    @Override
    public int rank() {
        return dims().length;
    }

    @Override
    public Tensor clone() {
        if(isScalar()){
            return Core.tensor(getScalar());
        }
        else {
            Tensor res = new Tensor(dims());

            for (int i = 0; i < getLength(); i++) {
                res.set(get(i).clone(), i);
            }
            return res;
        }
    }

    public static boolean dimsEqual(Tensor t1, Tensor t2) {
        return Arrays.equals(t1.dims(), t2.dims());
    }

    @Override
    public Iterator<Tensor> iterator() {
        return new MyIterator();
    }

    private class MyIterator implements Iterator<Tensor>{
        private int cursor = 0;
        @Override
        public boolean hasNext() {
            return cursor != getLength();
        }
        @Override
        public Tensor next() {
            Tensor nextItem = get(cursor);
            cursor++;
            return nextItem;
        }
    }

    /* AUTO GRADIENT START */

    /**
     *  Autograd
     * */
    private AutoGrad autoGrad; // always declared, but may not contain a gradient

    private boolean requires_grad = false;
    public Tensor requires_grad(boolean requires_grad){
        this.requires_grad = requires_grad;
        return this;
    }
    public boolean isRequires_grad(){
        return requires_grad;
    }

    private void setAutoGrad(AutoGrad autoGrad){
        this.autoGrad = autoGrad;
    }
    public AutoGrad getAutoGrad(){
        return autoGrad;
    }

    @Override
    public void _backward_() {
        if(!requires_grad)
            throwError("Tensor do not requires gradient");
        autoGrad._backward_();
    }

    @Override
    public Tensor getGrad() {
        return autoGrad.getGrad();
    }
    /* AUTO GRADIENT END */


    /* CORE OPERATIONS START */
    public Tensor add(Tensor other){
        return Core.sum(this, other, requires_grad);
    }
    public Tensor sub(Tensor other){
        return Core.sum(this, other, requires_grad);
    }
    public Tensor mul(Tensor other){
        return Core.sum(this, other, requires_grad);
    }
    public Tensor div(Tensor other){
        return Core.sum(this, other, requires_grad);
    }
    public Tensor dot(Tensor other){
        return Core.dot(this, other, requires_grad);
    }

    public Tensor conv(Tensor kernel){
        return Core.conv(this, kernel, requires_grad);
    }

    public Tensor tr(){
        return Core.tr(this);
    }
    public Tensor neg(){
        return Core.neg(this, requires_grad);
    }
    public Tensor mirror(){
        return Core.mirror(this);
    }
    public Tensor rotate(int angle){
        return  Core.rotate(this, angle);
    }
    /* CORE OPERATIONS END */

}
