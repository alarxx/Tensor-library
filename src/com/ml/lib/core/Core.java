package com.ml.lib.core;

import com.ml.lib.linear_algebra.operations.Conv;
import com.ml.lib.linear_algebra.operations.MatMul;
import com.ml.lib.linear_algebra.operations.morph_transform.Dilation;
import com.ml.lib.linear_algebra.operations.morph_transform.Erosion;
import com.ml.lib.linear_algebra.operations.self_operation.*;
import com.ml.lib.linear_algebra.operations.elementary.Div;
import com.ml.lib.linear_algebra.operations.elementary.Mul;
import com.ml.lib.linear_algebra.operations.elementary.Sub;
import com.ml.lib.linear_algebra.operations.elementary.Sum;

import com.ml.lib.tensor.Tensor;
import com.ml.lib.autograd.AutoGrad;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.ml.lib.tensor.Tensor.tensor;

public class Core {
    public static void throwError(String error){
        System.out.println(error);
        int a = 8 / 0;
    }



    public static boolean isVector(Tensor t){
        return t.rank() == 1;
    }
    public static boolean isColVector(Tensor t){
        return t.rank() == 2 && t.dims()[1] == 1;
    }
    public static boolean isRowVector(Tensor t){
        return t.rank() == 2 && t.dims()[0] == 1;
    }
    public static boolean isMatrix(Tensor t){
        return t.rank() == 2;
    }


    public static boolean dimsEqual(Tensor t1, Tensor t2) {
        return Arrays.equals(t1.dims(), t2.dims());
    }

    public static int[] reverse(int[] arr){
        int[] res = new int[arr.length];
        for(int i=0; i<arr.length; i++)
            res[i] = arr[arr.length - 1 - i];
        return res;
    }

    public static int numberOfElements(int [] dims){
        int res = 1;
        for(int i=0; i< dims.length; i++){
            res *= dims[i];
        }
        return res;
    }
    public static int numberOfElements(int [] dims, int rank){
        int res = 1;
        for(int i=0; i< dims.length - rank; i++){
            res *= dims[i];
        }
        return res;
    }




    /** очень долгие методы, но рабочие */
    public static Tensor castToDims(Tensor tensor, int ... dims){
        if(numberOfElements(tensor.dims()) != numberOfElements(dims)){
            throwError("Can not cast tensors to dims");
        }

        Tensor resultTensor = new Tensor(dims);

        List<Tensor> list = allTensorsOfRank(tensor, 0);

        castToDims(resultTensor, list);

        return resultTensor;
    }
    private static void castToDims(Tensor tensor, List<Tensor> list){
        if(tensor.isScalar()){
            tensor.setScalar(list.get(0).getScalar());
            list.remove(0);
        }
        else {
            for(Tensor t: tensor){
                castToDims(t, list);
            }
        }
    }

    public static Tensor firstTensorOfRank(Tensor tensor, int rank){
        if(tensor.rank() == rank){
            return  tensor;
        }
        else {
            return firstTensorOfRank(tensor.get(0), rank);
        }
    }

    public static List<Tensor> allTensorsOfRank(Tensor tensor, int rank){
        List<Tensor> list = new ArrayList<>();

        allTensorsOfRank(list, tensor, rank);

        return list;
    }
    private static void allTensorsOfRank(List<Tensor> list, Tensor tensor, int rank){
        if(tensor.rank() == rank){
            list.add(tensor);
        }
        else {
            for (Tensor t: tensor) {
                allTensorsOfRank(list, t, rank);
            }
        }
    }


    public static List<int[]> allKeysOfTensorsOfRank(Tensor tensor, int rank){
        List<Integer> indexes = new ArrayList<>();
        List<int[]> keys = new ArrayList<>();

        allKeysOfTensorsOfRank(keys, indexes, tensor, rank);

        return keys;
    }
    private static void allKeysOfTensorsOfRank(List<int[]> keys, List<Integer> indexes, Tensor tensor, int rank){
        if(tensor.rank() == rank){
            keys.add(list2arr(indexes));
        }
        else {
            List<Integer> newIndexes = new ArrayList<>(indexes);

            for (int i = 0; i < tensor.getLength(); i++) {
                Tensor t = tensor.get(i);
                newIndexes.add(i);
                allKeysOfTensorsOfRank(keys, newIndexes, t, rank);
                newIndexes.remove(newIndexes.size()-1);
            }
        }
    }
    private static int[] list2arr(List<Integer> list){
        int[] res = new int[list.size()];
        for(int i = 0; i < res.length; i++){
            res[i] = list.get(i);
        }
        return res;
    }


    /**
     * IDK where it may be needed
     * */
    public static Tensor list2Tensor(List<Tensor> list){
        int[] dims = list.get(0).dims();

        int[] resDims = new int[dims.length+1];

        for(int i=1; i<dims.length+1; i++)
            resDims[i] = dims[i];

        resDims[0] = list.size();

        Tensor res = new Tensor(resDims);

        for(int i=0; i<list.size(); i++){
            res.set(list.get(i), i);
        }

        return res;
    }

    /**
     * For example, tensor of rank=2  to tensor of rank=4
     * [2, 2] => [1, 1, 2, 2]
     * */
    public static Tensor upTheRank(Tensor tensor, int rank, boolean clone){
        int delta = rank - tensor.rank();

        int[] dims = new int[rank];

        for(int i=0; i < delta; i++){
            dims[i] = 1;
        }
        for(int i=0; i<tensor.rank(); i++){
            dims[i + delta] = tensor.dims()[i];
        }

        Tensor result = new Tensor(dims);

        Tensor kek = result;
        for(int i=0; i<delta; i++)
            kek = kek.get(0);

        kek.set(clone ? tensor.clone() : tensor);

        return result;
    }


    /*
    *  Не ко всем операциям написал производные
    *  OPERATIONS
    * */

    public static Tensor sum(Tensor t1, Tensor t2, boolean requires_grad){
        return requires_grad ? AutoGrad.sum(t1, t2) : Sum.getInstance().apply(t1, t2);
    }
    public static Tensor sum(Tensor t1, Tensor t2){
        return sum(t1, t2, false);
    }

    public static Tensor sub(Tensor t1, Tensor t2, boolean requires_grad){
        return requires_grad ? AutoGrad.sub(t1, t2) : Sub.getInstance().apply(t1, t2);
    }
    public static Tensor sub(Tensor t1, Tensor t2){
        return sub(t1, t2, false);
    }

    public static Tensor mul(Tensor t1, Tensor t2, boolean requires_grad){
        return requires_grad ? AutoGrad.mul(t1, t2) : Mul.getInstance().apply(t1, t2);
    }
    public static Tensor mul(Tensor t1, Tensor t2) {
        return mul(t1, t2, false);
    }

    public static Tensor div(Tensor t1, Tensor t2, boolean requires_grad){
        return Div.getInstance().apply(t1, t2);
    }

    public static Tensor div(Tensor t1, Tensor t2){
        return div(t1, t2, false);
    }

    public static Tensor dot(Tensor t1, Tensor t2, boolean requires_grad){
        return requires_grad ? AutoGrad.dot(t1, t2) : MatMul.getInstance().apply(t1, t2);
    }
    public static Tensor dot(Tensor t1, Tensor t2){
        return dot(t1, t2, false);
    }

    public static Tensor neg(Tensor tensor, boolean requires_grad){
        return requires_grad ? AutoGrad.neg(tensor) : mul(tensor, tensor(-1f), false);
    }
    public static Tensor neg(Tensor t1){
        return neg(t1, false);
    }

    /*CONVOLUTIONS START*/
    /** Полная форма */
    public static Tensor conv(Tensor tensor, Tensor kernel, int step, Conv.Type type, boolean requires_grad){
        return requires_grad ? AutoGrad.conv(tensor, kernel) : new Conv(step, type).apply(tensor, kernel);
    }

    /** step = 1 */
    public static Tensor conv(Tensor tensor, Tensor kernel, Conv.Type type, boolean requires_grad){
        return conv(tensor, kernel, 1, type, requires_grad);
    }

    /** type = SUM */
    public static Tensor conv(Tensor tensor, Tensor kernel, int step, boolean requires_grad){
        return conv(tensor, kernel, step, Conv.Type.SUM, requires_grad);
    }

    /** step = 1,
     * type = SUM */
    public static Tensor conv(Tensor tensor, Tensor kernel, boolean requires_grad){
        return conv(tensor, kernel, 1, Conv.Type.SUM, requires_grad);
    }

    /** step = 1,
     * type = SUM,
     * requires_grad = false*/
    public static Tensor conv(Tensor t1, Tensor t2){
        return conv(t1, t2, 1, Conv.Type.SUM, false);
    }
    /*CONVOLUTIONS END*/

    public static Tensor tr(Tensor tensor){
        return Transposition.getInstance().apply(tensor);
    }
    public static Tensor mirror(Tensor tensor){
        return new Mirror().apply(tensor);
    }
    public static Tensor rotate(Tensor tensor, int angle){
        return new Rotate(angle).apply(tensor);
    }

    public static Tensor max(Tensor tensor, int rank){
        return new MaxOfRank(rank).apply(tensor);
    }

    public static Tensor pow(Tensor tensor, float pow){
        return new Pow(pow).apply(tensor);
    }

    public static Tensor erode(Tensor tensor, Tensor kernel){
        return new Erosion().apply(tensor, kernel);
    }
    public static Tensor dilate(Tensor tensor, Tensor kernel){
        return new Dilation().apply(tensor, kernel);
    }

    public static Tensor rand(Tensor tensor, float min, float max){
        return new Rand(min, max).apply(tensor);
    }
    public static Tensor rand(Tensor tensor){
        return new Rand().apply(tensor);
    }
}
