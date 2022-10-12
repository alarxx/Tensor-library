package sheet.generic_try.linear_algebra.array;

import sheet.generic_try.linear_algebra.interfaces.MultiArrayInterface;

/*
* Императивный класс, методы изменяют значения самого класса
* */

public class MultiArray<T> implements MultiArrayInterface<T> {
    public static void main(String[] args) {
//        MultiArray<Integer> arr = new MultiArray<>(3, 3);
        MultiArray arr = new MultiArray(1, 2, 3);
        arr.set(1f, 0, 0, 0);
        System.out.println(arr.get(0, 0, 0));
    }

    private int[] dims;
    private int amountOfElements;
    private Object[] array;

    // произведение предшествующих измерений. Например: [3, 3, 3] => [3, 9, 27], [3, 4, 5] => [3, 12, 60]
    private int[] productOfPreviousDims;

    public MultiArray(int ... dims){ // 3, 3, 3
        this.dims = dims;
        this.amountOfElements = amountOfElements();
        this.array = new Object[amountOfElements];
        this.productOfPreviousDims = productOfPreviousDims(dims);
    }

    @Override
    public void set(T item, int ... indexes) {
        array[indexIn(indexes)] = item;
    }
    @Override
    public T get(int ... indexes) {
        return (T) array[indexIn(indexes)];
    }

    @Override
    public boolean contains(Object o) {
        return false;
    }

    @Override
    public void fill(T item){
        for(int i=0; i<amountOfElements; i++){
            array[i] = item;
        }
    }
    @Override
    public void clear() {
        fill(null);
    }

    @Override
    public int[] indexOf(Object o) {
        return null;
    }

    @Override
    public int[] lastIndexOf(Object o) {
        return null;
    }

    @Override
    public int size() {
        return amountOfElements;
    }

    @Override
    public int[] dims() {
        return this.dims;
    }

    @Override
    public boolean dimsEqual(int ... indexes){
        if(indexes.length != dims.length){
            System.out.println("indexes.length != dims.length");
            return false;
        }
        for(int i=0; i<dims.length; i++) {
            if (dims[i] <= indexes[i]) {
                System.out.println("Index out of bounds");
                return false;
            }
        }
        return true;
    }
    @Override
    public boolean dimsEqual(MultiArrayInterface array){
        return dimsEqual(array.dims());
    }

    @Override
    public String toString(){

        return null;
    }

    @Override
    public boolean equals(MultiArrayInterface array){
        // Здесь нужно прям поэлементно проверять
        //        return dimsConform(array.dims());
        return false;
    }

    private int amountOfElements(){
        int len = 1;
        for(int i=0; i<dims().length; i++){
            len *= dims[i];
        }
        return len;
    }

    private int indexIn(int ... indexes){
        if(!dimsEqual(indexes)){
            System.out.println("Measurements do not match");
            return -1;
        }
        if(indexes.length < 2){
            return indexes[0];
        }

        int res = 0;

        for(int i=indexes.length-1; i>1; i--){
            int depth = indexes[i];
            res += depth * productOfPreviousDims[i-1];
        }
        // Матрица расчитывается не так потому, что сначала идет row, потом col
        res += inMatrix(indexes);

        return res;
    }

    private int inMatrix(int ... indexes){
        int row = indexes[0];

        int col = indexes[1];
        int cols = dims[1];

        return cols * row + col;
    }

    private int[] productOfPreviousDims(int ... dims){
        int[] arr = new int[dims.length];
        arr[0] = dims[0];
        for(int i=1; i<arr.length; i++){
            int res = dims[i] * arr[i-1];
            arr[i] = res;
        }
        return arr;
    }

    @Override
    public Object[] __array(){
        return array;
    }
}
