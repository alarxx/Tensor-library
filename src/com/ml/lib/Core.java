package com.ml.lib;

import com.ml.lib.linear_algebra.operations.MatMul;
import com.ml.lib.linear_algebra.operations.Transposition;
import com.ml.lib.linear_algebra.operations.elementary.Div;
import com.ml.lib.linear_algebra.operations.elementary.Mul;
import com.ml.lib.linear_algebra.operations.elementary.Sub;
import com.ml.lib.linear_algebra.operations.elementary.Sum;

import com.ml.lib.tensor.Tensor;
import com.ml.lib.autograd.AutoGrad;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

    public static int numberOfElements(int ... dims){
        int res = 1;
        for(int i=0; i< dims.length; i++){
            res *= dims[i];
        }
        return res;
    }


    public static Tensor tensor(float scalar){
        return new Tensor().setScalar(scalar);
    }
    public static Tensor tensor(float[] vector){
        Tensor tensor = new Tensor(vector.length);

        for(int i=0; i<vector.length; i++){
            tensor.get(i).setScalar(vector[i]);
        }

        return tensor;
    }
    public static Tensor tensor(float[][] matrix){
        int     rows = matrix.length,
                cols = matrix[0].length;

        Tensor tensor = new Tensor(rows, cols);

        for(int i=0; i<rows; i++){
            tensor.set(tensor(matrix[i]), i);
        }

        return tensor;
    }
    public static Tensor tensor(float[][][] image){
        int     rows = image[0].length,
                cols = image[0][0].length,
                channels = image.length;

        Tensor tensor = new Tensor(channels, rows, cols);

        for(int i=0; i<channels; i++){
            tensor.set(tensor(image[i]), i);
        }

        return tensor;
    }
    public static Tensor tensor(float[][][][] array4){
        int     rows = array4[0][0].length,
                cols = array4[0][0][0].length,
                channels = array4[0].length,
                d = array4.length;

        Tensor tensor = new Tensor(d, channels, rows, cols);

        for(int i=0; i<d; i++){
            tensor.set(tensor(array4[i]), i);
        }

        return tensor;
    }


    /** очень долгие методы, но рабочие */
    public static Tensor castToDims(Tensor tensor, int ... dims){
        if(numberOfElements(tensor.dims()) != numberOfElements(dims)){
            throwError("Can not cast tensors to dims");
        }

        Tensor resultTensor = new Tensor(dims);

        List<Tensor> list = allTensorOfRank(tensor, 0);

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

    public static List<Tensor> allTensorOfRank(Tensor tensor, int rank){
        List<Tensor> list = new ArrayList<>();

        allTensorOfRank(list, tensor, rank);

        return list;
    }
    private static void allTensorOfRank(List<Tensor> list, Tensor tensor, int rank){
        if(tensor.rank() == rank){
            list.add(tensor);
        }
        else {
            for (Tensor t: tensor) {
                allTensorOfRank(list, t, rank);
            }
        }
    }


    public static List<int[]> allKeysOfTensorOfRank(Tensor tensor, int rank){
        List<Integer> indexes = new ArrayList<>();
        List<int[]> keys = new ArrayList<>();

        allKeysOfTensorOfRank(keys, indexes, tensor, rank);

        return keys;
    }
    private static void allKeysOfTensorOfRank(List<int[]> keys, List<Integer> indexes, Tensor tensor, int rank){
        if(tensor.rank() == rank){
            keys.add(list2arr(indexes));
        }
        else {
            List<Integer> newIndexes = new ArrayList<>(indexes);

            for (int i = 0; i < tensor.getLength(); i++) {
                Tensor t = tensor.get(i);
                newIndexes.add(i);
                allKeysOfTensorOfRank(keys, newIndexes, t, rank);
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
    *  FUNCTIONS
    * */

    public static Tensor sum(Tensor t1, Tensor t2, boolean grad){
        return grad ? AutoGrad.sum(t1, t2) : Sum.getInstance().apply(t1, t2);
    }
    public static Tensor sum(Tensor t1, Tensor t2){
        return sum(t1, t2, false);
    }

    public static Tensor sub(Tensor t1, Tensor t2, boolean grad){
        return grad ? AutoGrad.sub(t1, t2) : Sub.getInstance().apply(t1, t2);
    }
    public static Tensor sub(Tensor t1, Tensor t2){
        return sub(t1, t2, false);
    }

    public static Tensor mul(Tensor t1, Tensor t2, boolean grad){
        return grad ? AutoGrad.mul(t1, t2) : Mul.getInstance().apply(t1, t2);
    }
    public static Tensor mul(Tensor t1, Tensor t2){
        return mul(t1, t2, false);
    }

    public static Tensor div(Tensor t1, Tensor t2){
        return Div.getInstance().apply(t1, t2);
    }


    public static Tensor dot(Tensor t1, Tensor t2, boolean grad){
        return grad ? AutoGrad.dot(t1, t2) : MatMul.getInstance().apply(t1, t2);
    }
    public static Tensor dot(Tensor t1, Tensor t2){
        return dot(t1, t2, false);
    }


    public static Tensor neg(Tensor tensor){
        return mul(tensor, tensor(-1f));
    }

    /*
    * OPERATIONS
    * */
    public static Tensor tr(Tensor tensor){
        return Transposition.getInstance().apply(tensor);
    }

    public static Tensor conv(Tensor tensor, Tensor kernel){
        return Conv.getInstance().apply(tensor, kernel);
    }
}
