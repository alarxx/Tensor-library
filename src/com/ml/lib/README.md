# Tensor

## Tensor - array of Tensors, except rank-0 Tensor (scalar)
```
[Tensor, Tensor, Tensor, Tensor]  
   |       |       |       |  
   |       |     [...]   [...]  
   |  [Tensor, Tensor, ...]  
[Tensor, Tensor...]  
```
---
### Generics
Не получается писать дженерики.  
Есть проблема с дженериками в функциях, например:  
как понимать какого типа будет результирующая матрица;  
и скорость с ними будет меньше... хотя я и не претендую на скорость.

В этой реализации, как и в JS, все числа(скаляры) - float значения (поменял на double!).

---
### Scalars
Tensor - scalar, только если dims = [],  
но [1], [1, 1], [1, 1, 1]... не будут являться скалярами.

---
### Nulls
В первом варианте объявление Tensor-а - сложная операция,  
при создании обходится каждый элемент.  
Хотелось сделать так, чтобы если тензор под индексом не  
затрагивается, то по умолчанию он null.  
Для этого надо было было добавить проверку на null в get/set, fill, toString,  
а методы getScalar/setScalar не нуждаются в такой проверке:  
new Tensor().getScalar()/setScalar(),  
либо в связке с get - tensor.get(0).getScalar()/setScalar()

---
### Example of usage:
```
int     rows = 2, 
        cols = 3, 
        channels = 3;

Tensor t = new Tensor(channels, rows, cols);

for(int d=0, v=0; d<channels; d++){
    for(int r=0; r<rows; r++){
        for(int c=0; c<cols; c++, v++){
            t.set(tensor(v), d, r, c);
        }
    }
}

System.out.print(t);
```

### Added AutoGradient.
Любые сложные функции или операции можно представить   
в виде цепочки преобразований Тензоров, в виде графа,   
хотелось бы бинарного дерева, но один тензор может участвовать в нескольких функциях,  
в таком случае градиенты этих преобразований складываются и назначаются ему.  
Каждой операцие нужно прописывать то, как будет распростанятся градиент назад.  
В классе OperationGrad мы прописываем forward и backward.  
Метод forward всегда использует пакет linear_algebra.


### Added Core methods in Tensor
( Which are just a wrapper around the methods of the core class )
* Operations-methods of Tensor are based on the Core class.
* Methods do not change the state of the Tensor, but return a new resulting Tensor.
#### About gradient
* A Tensor that requires a gradient results in a Tensor that also requires a gradient.
* Operations do not change the parent Tensors requires_grad state.
#### Recommendations
* I recommend using only Tensor methods without affecting the Core class,
  Because it's more intuitive and convenient.
#### Example:
```
Tensor a = Tensor.tensor(new double[][]{
            {1, 2, 3},
            {4, 5, 6}
        })
        .requires_grad(true);

Tensor b = Tensor.tensor(new double[][]{
            {2, 3},
            {4, 5},
            {6, 7}
        });

Tensor c = a.dot(b);
System.out.println("c:"+c);

c._backward_();

System.out.println("c_der:"  + c.getGrad());
System.out.println("a_der:"  + a.getGrad());
System.out.println("b_der:"  + b.getGrad());
```
