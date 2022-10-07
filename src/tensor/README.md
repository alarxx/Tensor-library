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

В этой реализации, как и в JS, все числа(скаляры) - float значения.  

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