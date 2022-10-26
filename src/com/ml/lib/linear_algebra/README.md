# Linear Algebra

---

## All operations and functions on Tensors

Идея в том, чтобы по тензорно применять какую-то операцию.  
Под-Тензоры мы вытаскиваем по определенному ранку и  
применяем над ней или над ними какую-то операцию.

```
[[], [], []] <= [] 
```
по тензорно применяем на каждый под-тензор.
в результате у нас должно быть 3 каких-то тензора.  
Например, по-элементное умножение вектора[1, 2, 3, 4, 5, 6, 7, 8] на другой вектор[1, 2, 3, 4]:  
```
[1, 2, 3, 4,     5, 6, 7, 8]
 |  |  |  |      |  |  |  |
[1, 2, 3, 4],   [1, 2, 3, 4]

=> [1, 4, 9, 16, 5, 12, 21, 32].dims = {1, 8}
```
В случае по тензорных операций идея та же.  

Основной метод сопостовитель (если хотите написать свою операцию): 
```
public Tensor apply(Tensor src1, Tensor src2) {
        setRanks(ranksToCorrelate(src1, src2));
        setResultDims(resultTensorsDims(src1, src2));

        Tensor result = new Tensor(getResultDims());

        List<Tensor>    lt1     = Core.allTensorOfRank(src1, getRanks()[0]),
                        lt2     = Core.allTensorOfRank(src2, getRanks()[1]);

        // Мы можем сказать, какой длины будет res_lt, но не знаем какие это ранки в случае если result.dims={1, 3, 5, 5}
        int resultRankCorrelate = countResultRank(lt1.size(), lt2.size());

        List<Tensor>    res_lt  = Core.allTensorOfRank(result, resultRankCorrelate); // какой ранк мы должны взять, чтобы все четко совпало.

        if(     getRanks()[0] != 0 &&
                getRanks()[1] != 0 &&
                lt1.size() % lt2.size() != 0
        ){
            throwError("Something with dims is wrong here");
        }

        // res_lt.size() should be equal to max(lt1.size(), lt2.size())
        for(int i = 0; i < res_lt.size(); i++) {
            Tensor output = operation(
                    lt1.get(i % lt1.size()),
                    lt2.get(i % lt2.size())
            );

            res_lt.get(i).set(output);
        }

        return result;
}
```

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

