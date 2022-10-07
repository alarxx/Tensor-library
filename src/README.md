# Core

---

#### Чтобы не углубляться в классовую реализацию,  
#### можно пользоваться статичными методами из класса Core.  


#### Tensor declaration:
```
Tensor tensor = Core.tensor(3, 10, 10); 
```

#### Tensor element by element multiplication:
```
Tensor another = Core.tensor(1, 10, 10); 
Tensor resultOfMult = Core.mul(tensor, another);
```

#### AutoGrad example: 
```
Tensor a = Core.tensor(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });

Tensor b = Core.tensor(new float[][]{
        {2, 3},
        {4, 5},
        {6, 7}
});

Tensor c = Core.dot(a, b, true);
System.out.println(c);

c._backward_();

System.out.println("c_der:"  + c.getGrad());
System.out.println("a_der:"  + a.getGrad());
System.out.println("b_der:"  + b.getGrad());
```