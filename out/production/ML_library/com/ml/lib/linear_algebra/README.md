# Linear Algebra

---

## All operations and functions on Tensors

Идея в том, чтобы по тензорно применять какую-то операцию.  
Под-Тензоры мы вытаскиваем по определенному ранку и  
применяем над ней или над ними какую-то операцию.

Если это операция над тензором (транспонирование),  
Если это операция с двумя тензорами (умножение, поэлементные операции),  
Разницы ведь особой нет в определении...  
В одном случае мы результируем используя только один Тензор,  
В другом случае мы результируем используя два Тензора.
  
---
### Example of usage:
```
MatMul matMul = new MatMul();

float[][][] mat1 = new float[][][]{
        {
                {1, 2, 3},
                {4, 5, 6}
        },
        {
                {7, 8, 9},
                {10, 11, 12}
        }
};
Tensor matT1 = tensor(mat1);

float[][][] mat2 = new float[][][]{
        {
                {1, 2},
                {3, 4},
                {5, 6}
        },
        {
                {7, 8},
                {9, 10},
                {11, 12}
        }
};

Tensor matT2 = tensor(mat2);

Tensor result = matMul.apply(matT1, matT2);

System.out.println(matT1);
System.out.println(matT2);
System.out.println(result);
```

