package sheet.generic_try.linear_algebra.interfaces;

public interface MultiArrayInterface<T> {
    int[] dims();
    boolean contains(T o);

    void set(T item, int... indexes);

    T get(int ... indexes);

    void fill(T item);

    void clear();

    int[] indexOf(Object o);
    int[] lastIndexOf(Object o);

    int size();

    boolean equals(MultiArrayInterface array);
    boolean dimsEqual(int ... dims);
    boolean dimsEqual(MultiArrayInterface array);

    Object[] __array();
}
