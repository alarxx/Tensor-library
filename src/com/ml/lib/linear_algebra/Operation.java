package com.ml.lib.linear_algebra;

import com.ml.lib.Core;
import com.ml.lib.tensor.Tensor;

import java.util.List;

import static com.ml.lib.Core.throwError;

/**
 *  Вытаскивает под-тензоры ранка, опеределенного в rankToCorrelate и передает в operation;
 *  Значение полученное из operation назначается в Tensor result(resultDims());
 * */

public abstract class Operation implements com.ml.lib.interfaces.Operation {
    /**rank.length either 1 or 2*/
    private int[] ranks;

    private int[] resultDims;

    /**
     * length either 1 or 2.
     * <p>
     * Tensors of this rank will be passed to the operation.
     * <p>
     * In arguments original tensors. The method must not change the state of the argument.
     *
     * @param src2 may be null
     * */
    abstract protected int[] ranksToCorrelate(Tensor src1, Tensor src2);

    /**
     * Define resulting dims.
     * <p>
     * For example, the dimensions of the resulting matrix
     * during matrix multiplication or convolution may be different.
     * <p>
     * If it is necessary to somehow process each case from the
     * possible ranks, you can use getRank() method.
     * <p>
     * In arguments original tensors. The method must not change the state of the argument.

     * @param src2 may be null
     * */
    abstract protected int[] resultTensorsDims(Tensor src1, Tensor src2);

    /**
     * All tensors of a certain ranks are transferred as an argument one by one.
     * <p>
     * In arguments original tensors. The method must not change the state of the argument.
     * <p>
     * All tensors of a certain rank are transferred as an argument one by one.
     *
     * @param t2 may be null
     * */
    abstract protected Tensor operation(Tensor t1, Tensor t2);


    /*
     * Можно было бы закрывать закрывать метод из под-классов,
     * Я бы закрывал их в зависимости от длины ranks.length
     * */


    @Override
    public Tensor apply(Tensor src1, Tensor src2) {

        setRanks(ranksToCorrelate(src1, src2));
        if(getRanks().length != 2)
            throwError("something wrong");

        setResultDims(resultTensorsDims(src1, src2));

        Tensor result = new Tensor(getResultDims());

        List<Tensor>    lt1     = Core.allTensorOfRank(src1, getRanks()[0]),
                        lt2     = Core.allTensorOfRank(src2, getRanks()[1]);

        int resultRankCorrelate = countResultRank(lt1.size(), lt2.size());

        List<Tensor>    res_lt  = Core.allTensorOfRank(result, resultRankCorrelate); // какой ранк мы должны взять, чтобы все четко совпало.

        if(     getRanks()[0] != 0 &&
                getRanks()[1] != 0 &&
                lt1.size() % lt2.size() != 0
        ){
            throwError("Something with dims is wrong here");
        }

        for(int i = 0; i < res_lt.size(); i++) {
            Tensor output = operation(
                    lt1.get(i % lt1.size()),
                    lt2.get(i % lt2.size())
            );

            res_lt.get(i).set(output);
        }

        return result;
    }

    @Override
    public Tensor apply(Tensor src) {

        setRanks(ranksToCorrelate(src, null));
        if(getRanks().length != 1)
            throwError("something wrong");

        setResultDims(resultTensorsDims(src, null));

        Tensor result = new Tensor(getResultDims());

        List<Tensor>    lt1     = Core.allTensorOfRank(src, getRanks()[0]);

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

    protected int[] getRanks() {
        return ranks;
    }

    protected void setRanks(int[] ranks) {
        this.ranks = ranks;
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
