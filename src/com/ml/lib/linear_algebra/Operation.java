package com.ml.lib.linear_algebra;

import com.ml.lib.core.Core;
import com.ml.lib.tensor.Tensor;
import org.w3c.dom.ls.LSOutput;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.List;

import static com.ml.lib.core.Core.throwError;

/**
 *  Вытаскивает под-тензоры ранка, опеределенного в rankToCorrelate и передает в operation;
 *  Значение полученное из operation назначается в Tensor result(resultDims());
 * */

public abstract class Operation implements com.ml.lib.interfaces.Operation {
    private int[] ranks; // length = 3

    private int[] resultDims;

    public Operation(){}

    /**
     * The length of the returned array is always 3.
     * <p>
     * Tensors of this ranks will be passed to the operation.
     * <p>
     * In arguments original tensors. The method must not change the state of the argument.
     *
     * @param src2 may be null
     * */
    abstract protected int[] ranksToCorrelate(Tensor src1, Tensor src2);


    /**
     * Define resulting dims. Dims object must be independent.
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


    @Override
    public Tensor apply(Tensor src1, Tensor src2) {
        setRanks(ranksToCorrelate(src1, src2));

        setResultDims(resultTensorsDims(src1, src2));

        Tensor result = new Tensor(getResultDims());

        List<Tensor>    lt1     = Core.allTensorsOfRank(src1, getRanks()[0]),
                        lt2     = Core.allTensorsOfRank(src2, getRanks()[1]),
                        res_lt  = Core.allTensorsOfRank(result, getRanks()[2]); // какой ранк мы должны взять, чтобы все четко совпало.

        // Если не получается сопоставить количества входных тензоров
        if(     getRanks()[0] != 0 &&
                getRanks()[1] != 0 &&
                lt1.size() % lt2.size() != 0 &&
                lt2.size() % lt1.size() != 0
        ){
            throwError("Something with dims is wrong here");
        }

        // Если не получается сопоставить с количеством результирующих тензоров
        if(lt1.size() != res_lt.size() && lt2.size() != res_lt.size()){
            throwError("Something with dims is wrong here");
        }

        for(int i = 0; i < res_lt.size(); i++) {
            Tensor output = operation(
                    lt1.get(i % lt1.size()),
                    lt2.get(i % lt2.size())
            );
//            System.out.println("operation:" + output);
//            System.out.println("res:" + res_lt.get(i));
            res_lt.get(i).set(output);
        }

        return result;
    }


    @Override
    public Tensor apply(Tensor src) {
        setRanks(ranksToCorrelate(src, null));

        setResultDims(resultTensorsDims(src, null));

        Tensor result = new Tensor(getResultDims());

        List<Tensor>    lt1     = Core.allTensorsOfRank(src, getRanks()[0]),
                        res_lt  = Core.allTensorsOfRank(result, getRanks()[2]);

        if(lt1.size() != res_lt.size()){
            throwError("Something with dims is wrong here");
        }

        for(int i = 0; i < res_lt.size(); i++) {
            Tensor output = this.operation(
                    lt1.get(i % lt1.size()),
                    null
            );

            res_lt.get(i).set(output);
        }

        return result;
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

}
