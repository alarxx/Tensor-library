package sheet.generic_try.linear_algebra;

import sheet.generic_try.linear_algebra.array.MultiArray;

public class Tensor<T> extends MultiArray<T> {
    public Tensor(int ... dims){
        super(dims);
    }
}
