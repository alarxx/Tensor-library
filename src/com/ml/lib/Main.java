package com.ml.lib;

import com.ml.lib.core.Core;
import com.ml.lib.tensor.Tensor;

import java.time.Year;

import static com.ml.lib.tensor.Tensor.tensor;

public class Main {
    public static void main(String[] args) {
        Tensor t1 = new Tensor(1, 1);
        Tensor t2 = new Tensor();
        System.out.println(Core.dimsEqual(t1, t2));
    }
}