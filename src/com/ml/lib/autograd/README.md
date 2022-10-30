# Auto Gradient

## Определение того, как передается градиент назад по функции
## Пользуемся методами из Linear Algebra

Любые сложные функции или операции можно представить   
в виде цепочки преобразований Тензоров, в виде графа,   
хотелось бы бинарного дерева, но один тензор может участвовать в нескольких функциях,  
в таком случае градиенты этих преобразований складываются и назначаются ему.  
Каждой операцие нужно прописывать то, как будет распростанятся градиент назад.  
В классе Method мы прописываем forward и backward.  
Метод forward всегда использует пакет linear_algebra.

define function y:  
y(x) = f ( g(x) )  
or same as  
x -> g -> f -> y

Chain rule:  
y ' (x) = f ' ( g(x) ) * g ' (x)

if  
f(g) = g * 5,  
g(x) = x * 2,  
then:  
y'(y) = 1  
y'(g) = f'(g) = 5  
y'(x) = f'(g) * 2 = 10  

Для двух переменных то же самое, берем за константу второую переменную, если  
диференциируем по первой, и наоборот.

AutoGrad добавляет и расчитывает градиент к обычным методам преобразования  
Тензоров из папки linear_algebra. 

Чтобы создать свою функцию нужно создать класс OperationGrad и определить в нем,
Пользоваться только  методами Core.
1) как работает само преобразование ( forward ).
2) как передается градиент назад ( backward ).

#### AutoGrad Example:
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

Tensor c = a.getAutoGrad()._method_(new MatMul(), b);
System.out.println(c);

c._backward_();

System.out.println("c_der:"  + c.getGrad());
System.out.println("a_der:"  + a.getGrad());
System.out.println("b_der:"  + b.getGrad());
```
#### grad examples:
```
Y = 3 * 2 * 5
2 - variable
Y'(2) = 15 an so on
```

```
Y = (3 + 7) * 5 = sc * 5
C' = 0 - производная константы
x' = 1 - производная переменной
Y'(x=3) = ((x + 7) * 5)' = (x + 7)' * 5 + (x + 7) * 5' = 5x' = 5
Y'(5) = sc * 5 = sc
```

#### An example that fully reflects the essence of autograd:
```
Tensor s3 = tensor(3)
        .requires_grad(true);

Tensor s7 = tensor(7)
        .requires_grad(true);

Tensor s5 = tensor(5)
        .requires_grad(true);

Tensor sc1 = s3.mul(s7);
Tensor sc2 = s7.mul(s5);

Tensor Y = sc1.add(tensor(1f)); 
// Расчитывается градиент
Y._backward_();

Y = sc1.add(sc2);
// Прошлый градиент сбрасывается и расчитывается новый
Y._backward_(); 

System.out.println("Y: " + Y.getGrad()); // 1
System.out.println("sc1: " + sc1.getGrad()); // 1
System.out.println("sc2: " + sc2.getGrad()); // 1
System.out.println("s3: " + s3.getGrad()); // 7
System.out.println("s7: " + s7.getGrad()); // 8
System.out.println("s5: " + s5.getGrad()); // 7
```