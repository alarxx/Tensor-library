package com.ml.lib.linear_algebra;

import com.ml.lib.Core;
import com.ml.lib.tensor.Tensor;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.List;

import static com.ml.lib.Core.throwError;

/**
 *  Вытаскивает под-тензоры ранка, опеределенного в rankToCorrelate и передает в operation;
 *  Значение полученное из operation назначается в Tensor result(resultDims());
 * */

public abstract class Operation implements com.ml.lib.interfaces.Operation {

    private int[] rank;
    private int[] resultDims;

    /**
     * Tensors of this rank will be passed to the operation.
     * В function будут передаваться все тензоры этого ранка.
     * В аргументах оригинальные тензоры
     * @param src2 may be null
     * */
    abstract protected int[] ranksToCorrelate(Tensor src1, Tensor src2);

    /**
     * Define resulting dims.
     * Измерения результирующего тензора.
     * Нужно как-то обработать каждый кейс из возможных ранков.
     * В аргументах оригинальные тензоры.
     * Писал в основном с расчетом на измение размерностей матрицы при матричном умножении.
     * @param src2 may be null
     * */
    abstract protected int[] resultTensorsDims(Tensor src1, Tensor src2);

    /**
     * Сопоставляются тензоры определенных одинаковых ранков.
     * Метод не должен изменять состояние аргумента.
     * All tensors of a certain rank are transferred as an argument one by one.
     * @param t2 may be null
     * */
    abstract protected Tensor operation(Tensor t1, Tensor t2);


    /**
     * Можно было бы закрывать закрывать метод из под-классов...
     * */


    @Override
    public Tensor apply(Tensor src1, Tensor src2) {
        setRank(ranksToCorrelate(src1, src2));
        setResultDims(resultTensorsDims(src1, src2));

        Tensor result = new Tensor(getResultDims());

        List<Tensor>    lt1     = Core.allTensorOfRank(src1, getRank()[0]),
                        lt2     = Core.allTensorOfRank(src2, getRank()[1]);

        int resultRankCorrelate = countResultRank(lt1.size(), lt2.size());

        List<Tensor>    res_lt  = Core.allTensorOfRank(result, resultRankCorrelate); // какой ранк мы должны взять, чтобы все четко совпало.

        if(     getRank()[0] != 0 &&
                getRank()[1] != 0 &&
                lt1.size() % lt2.size() != 0
        ){
            throwError("Something with dims is wrong here");
        }

        for(int i = 0; i < res_lt.size(); i++) {
            Tensor output = operation(
                    lt1.get(i % lt1.size()),
                    lt2.get(i % lt2.size())
            );

            System.out.println(res_lt.get(i));

            res_lt.get(i).set(output);
        }

        return result;
    }

    @Override
    public Tensor apply(Tensor src) {
        setRank(ranksToCorrelate(src, null));
        setResultDims(resultTensorsDims(src, null));

        Tensor result = new Tensor(getResultDims());

        List<Tensor>    lt1     = Core.allTensorOfRank(src, getRank()[0]);

        int resultRankCorrelate = countResultRank(lt1.size(), 0);

        List<Tensor>    res_lt  = Core.allTensorOfRank(result, resultRankCorrelate);

        for(int i = 0; i < res_lt.size(); i++) {
            Tensor output = this.operation(
                    lt1.get(i % lt1.size()),
                    null
            );

            res_lt.get(i).set(output);
        }

        return result;
    }

    private int countResultRank(int n1, int n2) {
        int number = Math.max(n1, n2);

        if(number == 1){
            return getResultDims().length;
        }

        // [3, 3] мы можем применить сюда максимум всего 9 операций,
        // мы можем применить сюда 3 или 9 операций, иначе не можем
        int number_counter = 1;

        for(int i=0; i < getResultDims().length; i++){
            number_counter *= resultDims[i];
            if(number == number_counter){
                System.out.println(getResultDims().length - i);
                return getResultDims().length - i - 1;
            }
        }

        throwError("Something with dims are wrong here");
        return -1;
    }

    protected int[] getRank() {
        return rank;
    }

    protected void setRank(int[] rank) {
        this.rank = rank;
    }

    protected int[] getResultDims() {
        return resultDims;
    }

    protected void setResultDims(int[] resultDims) {
        this.resultDims = resultDims;
    }

    protected int getResultRank(){
        return getResultDims().length;
    }
}
