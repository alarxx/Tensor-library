package com.ml.lib;

import com.ml.lib.core.Core;
import com.ml.lib.linear_algebra.operations.elementary.Mul;
import com.ml.lib.tensor.Tensor;

import java.time.Year;
import java.util.Arrays;

import static com.ml.lib.tensor.Tensor.tensor;

public class Main {
    public static void main(String[] args) {
        Tensor t = new Tensor(1, 1);
        t.get(0, 0).setScalar(3f);

        Tensor n = new Mul().apply(t, tensor(3f));

        System.out.println(Arrays.toString(n.dims()));
        System.out.println(n);
    }
}