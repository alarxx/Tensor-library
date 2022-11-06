package com.ml.lib.linear_algebra.operations.elementary;

import com.ml.lib.linear_algebra.Operation;
import com.ml.lib.tensor.Tensor;

import static com.ml.lib.core.Core.*;
import static com.ml.lib.tensor.Tensor.tensor;

/**
 *
 * Комент устарел. Оставлю на всякий случай.
 * Пока думаю, что хорошо будет работать Mat-scalar, Mat-mat, 3d-3d, 3d-2d, 3d-scalar,
 * Но на счет 4д не уверен. там могут быть траблы уже, вся ответственность на пользователе.
 *
 * Проблема в том, что я не проверяю количество матриц при сопоставлении, потому что
 * есть неочевидные случаи когда количество матриц совпадает, но измерения не совпадают и т.д.
 *
 * Надо бы по идее сделать не поэлементное умножение в матрицах, а сопоставлять
 * rowVector and colVector, дублировать как бы вектора. Так вроде делается в pytorch.
 * Но пока работает, только с матрицами одинаковых размерностей (mxn) and (mxn)
 *
 *
 * Вообще, если оставить конкретно в такой реализации, то можно и не писать реализацию суммирования матриц?
 * */
public abstract class ElementByElement extends Operation {

    /** Операции суммирования, вычитания, умножения, деления */
    public abstract float operation(float a, float b);

    @Override
    protected int[] ranksToCorrelate(Tensor src1, Tensor src2) {
        int[] ranks = new int[]{0, 0, 0};

        if(src1.rank() >= 2 && src2.rank() >= 2){
            ranks = new int[]{2, 2, 2};
        }

        return ranks;
    }


    @Override
    protected int[] resultTensorsDims(Tensor src1, Tensor src2) {
        int numOfElements1 = numberOfElements(src1.dims()),
                numOfElements2 = numberOfElements(src2.dims());
        if(getRanks()[0] == 0 && getRanks()[1] == 0){
            return numOfElements1 > numOfElements2 ? src1.dims().clone() : src2.dims().clone();
        }
        else {
            return numOfElements1 > numOfElements2 ? src1.dims().clone() : src2.dims().clone();
        }
    }


    @Override
    public Tensor operation(Tensor t1, Tensor t2) {
        if (getRanks()[0] == 0 && getRanks()[1] == 0) { // Это скаляры
            float value = operation(t1.getScalar(), t2.getScalar());
            return tensor(value);
        }
        else { // Это матрицы
            return matrices(t1, t2);
        }
    }

    private Tensor matrices(Tensor mat1, Tensor mat2){
        int     rows1 = mat1.dims()[0],
                cols1 = mat1.dims()[1],
                rows2 = mat2.dims()[0],
                cols2 = mat2.dims()[1];

        if (rows1 == rows2 && cols1 == cols2){
            return matricesEqual(mat1, mat2);
        }
        // second mat - scalar matrix like [[1f]] of dims [1, 1]
        else if (rows2 == 1 && cols2 == 1) {
            return secondMatIsScalar(mat1, mat2);
        }
        // first mat - scalar matrix like [[1f]] of dims [1, 1]
        else if (rows1 == 1 && cols1 == 1) {
            return secondMatIsScalar(mat2, mat1);
        }
        // second mat - colvec like [[1], [2], [3]]
        else if(cols2 == 1){
            return secondMatColVec(mat1, mat2);
        }
        // first mat - colvec like [[1], [2], [3]]
        else if(cols1 == 1){
            return secondMatColVec(mat2, mat1);
        }
        // row vector
        else if(rows2 == 1){
            return secondMatRowVec(mat1, mat2);
        }
        // row vector
        else if(rows1 == 1){
            return secondMatRowVec(mat2, mat1);
        }

        throwError("IDK WHY");
        return null;
    }

    private Tensor matricesEqual(Tensor mat1, Tensor mat2){
        int     rows = mat1.dims()[0],
                cols = mat1.dims()[1];

        Tensor result = new Tensor(rows, cols);

        for (int r = 0; r < rows; r++) {
            Tensor resultRow = result.get(r),
                    t1Row = mat1.get(r),
                    t2Row = mat2.get(r);

            for (int c = 0; c < cols; c++) {
                float value = operation(t1Row.get(c).getScalar(), t2Row.get(c).getScalar());
                resultRow.get(c).setScalar(value);
            }
        }
        return result;
    }

    private Tensor secondMatIsScalar(Tensor mat1, Tensor mat2){
        int     rows = mat1.dims()[0],
                cols = mat1.dims()[1];

        float scalar_matrix = mat2.get(0, 0).getScalar();
        Tensor result = new Tensor(rows, cols);

        for (int r = 0; r < rows; r++) {
            Tensor resultRow = result.get(r),
                    t1Row = mat1.get(r);

            for (int c = 0; c < cols; c++) {
                float value = operation(t1Row.get(c).getScalar(), scalar_matrix);
                resultRow.get(c).setScalar(value);
            }
        }
        return result;
    }

    private Tensor secondMatColVec(Tensor mat1, Tensor colvec){
        int     rows = mat1.dims()[0],
                cols = mat1.dims()[1],
                rows2 = colvec.dims()[0];

        if(rows % rows2 != 0){
            throwError("Not match");
        }

        Tensor result = new Tensor(rows, cols);

        for (int r = 0; r < rows; r++) {
            Tensor resultRow = result.get(r),
                    t1Row = mat1.get(r),
//                    t2Row = colvec.get(r % rows2), // это бы растягивало этот вектор, но это странно
                    t2Row = colvec.get(r);

            for (int c = 0; c < cols; c++) {
                float value = operation(t1Row.get(c).getScalar(), t2Row.get(0).getScalar());
                resultRow.get(c).setScalar(value);
            }
        }
        return result;

    }

    private Tensor secondMatRowVec(Tensor mat1, Tensor rowvec){

        int     rows = mat1.dims()[0],
                cols = mat1.dims()[1],
                cols2 = rowvec.dims()[1];

        if(cols % cols2 != 0){
            throwError("Not match");
        }

        rowvec = rowvec.get(0);

        Tensor result = new Tensor(rows, cols);

        for (int r = 0; r < rows; r++) {
            Tensor  resultRow = result.get(r),
                    t1Row = mat1.get(r);

            for (int c = 0; c < cols; c++) {
                float value = operation(t1Row.get(c).getScalar(), rowvec.get(c).getScalar()); // rowvec.get(c % cols2).getScalar()) это бы расстягивало вектор
                resultRow.get(c).setScalar(value);
            }
        }
        return result;

    }
}
