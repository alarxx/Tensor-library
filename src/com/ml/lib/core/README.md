# Core

---

#### Чтобы не углубляться в классовую реализацию,  
#### можно пользоваться статичными методами из класса Core.  


#### Tensor declaration:
```
Tensor tensor = new Tensor(3, 10, 10); 
```

#### Tensor element by element multiplication:
```
Tensor another = new Tensor(1, 10, 10); 
Tensor resultOfMult = Core.mul(tensor, another);
```

#### AutoGrad example:
If we use Core methods,  
then you can specify whether the gradient is needed for the result tensor.
```
Tensor a = tensor(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

Tensor b = tensor(new float[][]{
        {2, 3},
        {4, 5},
        {6, 7}
});

boolean gradientNeeded = true;
Tensor c = Core.dot(a, b, gradientNeeded);
System.out.println(c);

c._backward_();

System.out.println("c_der:"  + c.getGrad());
System.out.println("a_der:"  + a.getGrad());
System.out.println("b_der:"  + b.getGrad());
```

