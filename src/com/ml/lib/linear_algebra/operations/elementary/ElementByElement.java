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
        int[] rank = new int[]{0, 0};

        if(src1.rank() >= 2 && src2.rank() >= 2){
            rank = new int[]{2, 2};
        }

        return rank;
    }


    @Override
    protected int[] resultTensorsDims(Tensor src1, Tensor src2) {
        if(getRanks()[0] == 0 && getRanks()[1] == 0){
            if(numberOfElements(src1.dims()) > numberOfElements(src2.dims())){
                return src1.dims();
            }
            else{
                return src2.dims();
            }
        }
        else {
            int[]   dims1 = firstTensorOfRank(src1, 2).dims(),
                    dims2 = firstTensorOfRank(src2, 2).dims();

            int     rows = dims1[0],
                    cols = dims1[1],
                    rows2 = dims2[0],
                    cols2 = dims2[1];

            if(rows2!=1 && cols2!=1) {
                if (rows != rows2 || cols != cols2) {
                    throwError("Dimensions must be the same");
                }
            }

            return src1.dims();
        }
    }


    @Override
    public Tensor operation(Tensor t1, Tensor t2) {
        if (getRanks()[0] == 0 && getRanks()[1] == 0) {
            float value = operation(t1.getScalar(), t2.getScalar());
            return tensor(value);
        }
        else { // Это матрицы
            // Здесь нужно как-нибудь проверять и дополнять вектор до матрицы и тд????!!!!!!!!!!!!! Хотя в матеше так делать нельзя
            int     rows = t1.dims()[0],
                    cols = t1.dims()[1],
                    rows2 = t2.dims()[0],
                    cols2 = t2.dims()[1];

            if(rows2 == 1 && cols2 == 1){
                float scalar_matrix = t2.get(0, 0).getScalar();
                Tensor result = new Tensor(rows, cols);

                for(int r=0; r<rows; r++){
                    Tensor  resultRow = result.get(r),
                            t1Row = t1.get(r);

                    for(int c=0; c<cols; c++){
                        float value = operation(t1Row.get(c).getScalar(), scalar_matrix);
                        resultRow.get(c).setScalar(value);
                    }
                }
                return result;
            }

            if (rows != rows2 || cols != cols2)
                throwError("Wrong dimensions");

            Tensor result = new Tensor(rows, cols);

            for (int r = 0; r < rows; r++) {
                Tensor resultRow = result.get(r),
                        t1Row = t1.get(r),
                        t2Row = t2.get(r);

                for (int c = 0; c < cols; c++) {
                    float value = operation(t1Row.get(c).getScalar(), t2Row.get(c).getScalar());
                    resultRow.get(c).setScalar(value);
                }
            }

            return result;
        }
    }
}
