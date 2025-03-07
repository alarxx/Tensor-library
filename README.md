# Tensor-library (C++)

## Tensor 

Tensor is a recursive array of Tensors, except rank-0 Tensor (scalar)
```
[Tensor, Tensor, Tensor, Tensor]  
   |       |       |       |  
   |       |     [...]   [...]  
   |  [Tensor, Tensor, ...]  
[Tensor, Tensor...]  
```

---

## Features

### Tensors

#### Constructors

Tensor может быть только of arithmetic type.
Для проверки типов я использую [`concepts`](https://github.com/federico-busato/Modern-CPP-Programming), а не чистый SFINAE, поэтому target C++20.

Создать 3D tensor можно так, by default tensor of type double:
```c++
Tensor tensor(depth, rows, cols); // Tensor<double>, rank-3
```

You can specify type:
```c++
Tensor<int> tensor(depth, rows, cols); // specified tensor of type int
```

Integer dims only:
```c++
Tensor t(2, 2) // Ok, provided integers, creates 2x2 matrix
Tensor t(2., 2.) // Error due to incorrect type double
```

#### Scalar creation

Как создать scalar tensor:
```c++
Tensor sc = scalar(42); // Tensor<int>
```

Factory function scalar нужна потому что `Tensor t(42)` вызовет конструктор dims, который создаст вектор соответствующего размера, а `Tensor t = {42}` вызовет initializer_list конструктор, который создаст вектор с одним соответствующим элементом, но не скаляр!

У меня нет copy конструктора принимающего скаляр.
`Tensor t = 42` не сработает,
во-первых потому что у меня конструктор explicit,
во-вторых если бы он преобразовывался в `Tensor t(42)`, то такой конструктор конфликтовал бы с конструктором dims.
Но, scalar copy assignment operator сработает, если сначала создать пустой Tensor.

Под капотом `scalar(42)` работает так:
```c++
scalar(T value){
   Tensor<T> sc;
   sc = value;
   return sc;
}
```

#### Accessing elements

```c++
Tensor matrix(2, 2); // Tensor<double>
cout << matrix;
// {{0. 0.},
//  {0. 0.}}

// index operator matrix[i] returns tensor object

matrix[0][0] = 42; // scalar copy assignment = , implicit casting to double

double sc = matrix[0][0].value(); // get scalar value from tensor object

cout << matrix;
// {{42. 0.},
//  { 0. 0.}}
```

#### Initializer list

Создать tensor можно вручную, что реализовано с `initializer_list`:
```c++
Tensor vector = {1, 2, 3}; // 1D vector

Tensor matrix = { // 2D matrix
   {1, 2, 3},
   {4, 5, 6},
   {7, 8, 9}
};
```

Тут стоит отметить, что конструктор принимающий dims - `explicit`, то есть передавать мы можем только явно, например: `Tensor t(rows, cols)`.
Без `explicit` с `Tensor t = {rows, cols}` вызывал бы конструктор dims, а не `initializer_list`.

#### RAII and Rule of 5

Проект реализует [RAII](https://en.cppreference.com/w/cpp/language/raii)
i.e. Scope-Bound Resource Management
для управления выделением и освобождением памяти of raw recursive arrays.

Copy and move constructors:
```c++
Tensor vector = {1, 2, 3};

Tensor copied = vector;

Tensor stealed = std::move(vector);

cout << vector; // 0i
```

Copy and move assignments:
```c++
Tensor vector = {1, 2, 3};

Tensor copied(3); // sizes must  match
copied = vector;

cout << vector; // tensor<i>: {1i, 2i, 3i}

Tensor stealed(3);
stealed = std::move(vector);

cout << vector; // tensor<i>: 0i
```

#### Concat tensors

Тензоры одинаковой размерности можно объединять в один тензор на ранк выше:
```c++
Tensor v1 = {1, 2, 3};
Tensor v2 = {4, 5, 6};
Tensor v3 = {7, 8, 9};

Tensor copied(v1, v2, v3); // 1 time copy

Tensor stealed(std::move(v1), std::move(v2), std::move(v3)); // 0 copy, steals data
```

#### Elementwise multiplication

Unary multiplication:
```c++
Tensor v1 = {1, 2, 3};
Tensor v2 = {4, 5, 6};

v1 *= v2; // 0 copy, but may be overhead of recursion and

cout << v1; // {4, 10, 18}
cout << v2; // {4, 5, 6}
```

Binary multiplication:
```c++
Tensor v1 = {1, 2, 3};
Tensor v2 = {4, 5, 6};

Tensor mul = v1 * v2; // 1 copy + RVO

cout << v1; // {1, 2, 3}
cout << v2; // {4, 5, 6}
cout << mul; // {4, 10, 18}
```

#### Create tensor from any dimensional raw array

```c++
// so you have raw array of any dimensionality
int ma[2][3] = {
   {1, 2, 3},
   {1, 2, 3}
};

// you can pass dims directly like this, as rvalue:
Tensor matrix = from_array({2, 3}, ma); // Tensor<int> - deduction works

// or like this passing dims lvalue:
// std::vector dims = {2, 3};
// Tensor matrix = from_array(dims, ma);
// Это может быть полезно если в рантайме как-то пытаетесь преобразовывать, а не вручную

cout << matrix << endl;
/*
tensor(2)<i>:{
1i 2i 3i
1i 2i 3i
}
*/
```

#### Create tensor from STL vector

```c++
std::vector<std::vector<int>> mv = {
   {1, 2},
   {3, 4}
};

Tensor matrix = from_stl_vector(mv);

cout << matrix << endl;
/*
tensor(2)<i>:{
1i 2i
3i 4i
}
*/
```

#### Non-rectangular tensors

Tensor может быть не прямоугольным?
Это когда nested тензоры могут быть разного size.

Случай непрямоугольности я явно не запращею,
но я расчитывал на то, что tensor прямоугольный, поэтому нужно тестить.
При передаче размерности в контструктор создается прямоугольный tensor.

Я думаю, что не буду создавать явные проверки или дополнительную логику дополнения до прямоугольности с поиском самой длинной строки и тому подобное.

Просто то, что в `shape()` я буду считать размер по первым тензорам... ???

- В случае непрямоугольности копирование работает и не обрезает до прямоугольности.
- Непрямоугольность будет работать и с умножением.
- initializer_list конструкторы работают при передаче непрямоугольных листов

```c++
std::vector<std::vector<int>> mv = {
   {1, 2, 3},
   {3, 4}
};

// Конвертирует непрямоугольные векторы соответсвенно
Tensor matrix = from_stl_vector(mv);

// Копирование
Tensor matrix_copy = matrix; // constructor
matrix_copy = matrix; // assignment
cout << matrix_copy << endl;

// Получается непрямоугольность будет работать и с умножением.
matrix_copy *= matrix; // Unary elementwise
cout << matrix_copy << endl; // elements squared
/*
1 4 9
9 16
*/

Tensor mul = matrix * matrix_copy; // Binary elementwise
cout << mul << endl; // elements cubed
/*
1 8 27
27 64
*/
```

---

## Licence

Tensor-library is licensed under the terms of [MPL-2.0](https://mozilla.org/MPL/2.0/), which is simple and straightforward to use, allowing this project to be combined and distributed with any proprietary software, even with static linking. If you modify, only the originally covered files must remain under the same MPL-2.0.

License notice:
```
SPDX-License-Identifier: MPL-2.0
--------------------------------
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file,
You can obtain one at https://mozilla.org/MPL/2.0/.

This file is part of the Tensor-library:
https://github.com/alarxx/Tensor-library

Description: <Description>

Provided “as is”, without warranty of any kind.

Copyright © 2025 Alar Akilbekov. All rights reserved.

Third party copyrights are property of their respective owners.
```

---

Если вы собираетесь модифицировать MPL покрытые файлы (IANAL):
- Ваш fork проект = MPL Covered files + Not Covered files (1.7).
- MPL-2.0 is file-based weak copyleft действующий только на Covered файлы, то есть добавленные вами файлы и исполняемые файлы, полученные из объединенения с вашими, могут быть под любой лицензией (3.3).
- (но под copyleft могут подпадать и новые файлы в которых copy-paste-нули код из Covered) (1.7).
- Покрытыми лицензией (Covered) считаются файлы с license notice (e.g. .cpp, .hpp) и любые исполняемые виды этих файлов (e.g. .exe, .a, .so) (1.4).
- You may not remove license notices (3.4), как и в MIT, Apache, BSD (кроме 0BSD) etc.
- При распространении любой Covered файл должен быть доступен, но разрешено личное использование или только внутри организации (3.2).
- Если указан Exhibit B, то производную запрещается лицензировать с GPL.
- Contributor дает лицензию на любое использование конкретной интеллектуальной собственности (patent), которую он реализует в проекте (но не trademarks).

Эти разъяснения условий не меняют и не вносят новые юридические требования к MPL.

---

## Contact

Alar Akilbekov - alar.akilbekov@gmail.com

---

## References:
- Weidman, S. (2019). Deep learning from scratch: Building with Python from first principles (First edition). O’Reilly Media, Inc.
- Patterson, J., & Gibson, A. (2017). Deep learning: A practitioner’s approach (First edition). O’Reilly.
- Евгений Разинков. (2021). Лекции по Deep Learning. https://www.youtube.com/playlist?list=PL6-BrcpR2C5QrLMaIOstSxZp4RfhveDSP
- Raschka, S., Liu, Y., Mirjalili, V., & Dzhulgakov, D. (2022). Machine learning with PyTorch and Scikit-Learn: Develop machine learning and deep learning models with Python. Packt.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. The MIT press.
- Rashid, T. (2016). Make Your own neural network. CreateSpace Independent Publishing Platform.
