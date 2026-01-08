# Tensor-library (C++)

## Tensor 

Tensor is a multidimensional array implemented with mappings (basically, one large array), allowing fast access operations and suitable for SIMT operations.
```
Tensor = [batch * depth * rows * cols]
```

---

## Why c++mapping branch

Замена рекурсивной имплементации на хранение данных в одном длинном массиве.
Рекурсивная реализация по "Space Complexity" занимает в 3-6 раз больше.
Доступ к элементам в рекурсивной реализации needs to follow each pointer individually, resulting in many memory accesses,
в реализации с mapping-ом индексов only simple arithmetic and 1 memory access is performed, it is faster.
В рекурсивной имплементации O(rank) потому что для доступа к scalar-у идет рекурсивный проход по массивам rank раз, также кажется cost of access array operation высокий.
Хранение в длинном массиве должно быть удобным для SIMD/SIMT.

```c++
template <Arithmetic T = float>
class Tensor {
private:
   T * coeffs;
   int * dims;
   int rank;
public:
   using type = T;

   explicit Tensor() : rank(0), dims(nullptr), coeffs(new T[1]) {}

   template <Arithmetic U>
   friend Tensor<U> scalar(U value);

   explicit Tensor(int rank, int dims[]);

   explicit Tensor(std::integral auto ... args);

   Tensor(const INITIALIZER_LIST_1(T) list);
   Tensor(const INITIALIZER_LIST_2(T) list);
   Tensor(const INITIALIZER_LIST_3(T) list);
   Tensor(const INITIALIZER_LIST_4(T) list);

   ~Tensor();
   Tensor(const Tensor<T> & other);
   Tensor<T> & operator = (const Tensor<T> & other);
   Tensor(Tensor<T> && other);
   Tensor<T> & operator = (Tensor<T> && other); // tensor[index] = Tensor();

   inline T& get(std::integral auto ... args);

   std::string toString() const;

   template <Arithmetic U>
   friend std::ostream& operator<<(std::ostream& os, const Tensor<U>& tensor);
};
```

## TODO

2026-01-07
Need to do:
- [ ] Concat, it's better to left only concat friend function
- [ ] Tensor operataions: unary/binary (operator +-*/ overloading), matmul, ...
Weidman, S. (2019)
- [ ] AutoGrad
- [ ] DL classes

---

## Features

### Tensors

#### Constructors

Tensor может быть только of arithmetic type.
Для проверки типов я использую [`concepts`](https://en.cppreference.com/w/cpp/language/constraints), а не чистый SFINAE, поэтому target C++20.

Создать 3D tensor можно так, by default tensor of type float:
```c++
int depth = 2, rows = 3, cols = 3;
Tensor tensor(depth, rows, cols); // Tensor<float>, rank-3
cout << tensor;
/*
tensor(3D)<f>:
 0. 0. 0.
 0. 0. 0.
 0. 0. 0.

 0. 0. 0.
 0. 0. 0.
 0. 0. 0.
*/
```

You can specify type:
```c++
Tensor<int> tensor(depth, rows, cols); // specified tensor of type int
cout << tensor;
/*
tensor(3D)<i>:
 0 0 0
 0 0 0
 0 0 0

 0 0 0
 0 0 0
 0 0 0
*/
```

For dims allowed only int values:
```c++
double depth = 3.0, rows = 3.0, cols = 3.0;
Tensor<int> tensor(depth, rows, cols); // Error due to incorrect type double
```

#### Scalar creation

Как создать scalar tensor:
```c++
Tensor sc = scalar(42); // type deduction: Tensor<int>
```

"scalar" (factory function) нужна потому что `Tensor t(42)` вызовет конструктор dims, который создаст длинный вектор соответствующего размера, а `Tensor t = {42}` вызовет initializer_list конструктор, который создаст вектор с одним соответствующим элементом, но не скаляр!

У меня нет copy конструктора принимающего скаляр.
`Tensor t = 42` не сработает,
во-первых потому что у меня конструктор explicit,
во-вторых если бы он преобразовывался в `Tensor t(42)`, то такой конструктор конфликтовал бы с конструктором dims.
Можно создать scalar copy assignment operator `sc = 42`, но я считаю это плохим решением, так как он добавляет неявные абстракции.
Лучше явно писать `sc.get() = 42`.

Под капотом `scalar(42)` работает так:
```c++
scalar(T value){
   Tensor<U> tensor;
   *tensor.coeffs = value; // tensor[0] or tensor.get()
   return tensor; // RVO
}
```

#### Accessing elements

```c++
Tensor matrix(2, 2); // Tensor<float>
cout << matrix;
// {{0. 0.},
//  {0. 0.}}

matrix.get(0, 0) = 42; // assignment by reference, implicit casting to float

float sc = matrix.get(0, 0); // 42, get scalar value from tensor object

cout << matrix;
// {{42. 0.},
//  { 0. 0.}}
```

Would be nice to implement the idea to call `tensor[d][r][c]`.
Here is the problem with proxy "TensorView" which should not make copy, imo.
Also, I don't want to overcomplicate things.
```c++
class Tensor {
   ...
   inline Tensor& operator [] (const int index) {
      return coeffs[index]; // must return Tensor object
   }
}
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

// 3D, 4D as well
```

Тут стоит отметить, что конструктор принимающий dims - `explicit`, то есть передавать мы можем только явно, например: `Tensor t(rows, cols)`.
Без `explicit` с `Tensor t = {rows, cols}` вызывал бы конструктор dims, а не `initializer_list`.

#### RAII and Rule of 5

Проект реализует [RAII](https://en.cppreference.com/w/cpp/language/raii)
i.e. Scope-Bound Resource Management
для управления выделением и освобождением памяти of raw array.

Copy and move constructors:
```c++
Tensor vector = {1, 2, 3};

Tensor copied = vector; // O(N) copy

Tensor stealed = std::move(vector); // O(1)

cout << vector; // tensor(null)
```

Copy and move assignments:
```c++
Tensor vector = {1, 2, 3}; // Tensor<int>

Tensor copied; // doesn't require dims to match
copied = vector; // O(N) copy

cout << vector; // tensor<i>: {1i, 2i, 3i}

Tensor stealed;
stealed = std::move(vector); // O(1)

cout << vector; // tensor(null)
```

#### Concat tensors (not implemented in c++mappings branch)

Тензоры одинаковой размерности можно объединять в один тензор на ранк выше:
```c++
Tensor v1 = {1, 2, 3};
Tensor v2 = {4, 5, 6};
Tensor v3 = {7, 8, 9};

Tensor copied(v1, v2, v3); // 1 time copy

Tensor stealed(std::move(v1), std::move(v2), std::move(v3)); // 0 copy, steals data
```

#### Elementwise multiplication (not implemented in c++mappings branch)

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

#### Create tensor from any dimensional raw array (not implemented in c++mappings branch)

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

#### Create tensor from STL vector (not implemented in c++mappings branch)

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

#### Iterator (not implemented in c++mappings branch)

Iterator example:
```c++
Tensor tensor = {
   {1, 2, 3},
   {4, 5, 6},
   {7, 8, 9},
};

for(auto & t: tensor){
   std::cout << t << std::endl;
}
/*
   tensor(1)<i>:{
   1i 2i 3i
   }
   tensor(1)<i>:{
   4i 5i 6i
   }
   tensor(1)<i>:{
   7i 8i 9i
   }
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

Copyright © 2022-2026 Alar Akilbekov. All rights reserved.

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

C++:
- Modern C++: https://github.com/federico-busato/Modern-CPP-Programming
- Make: https://www3.ntu.edu.sg/home/ehchua/programming/cpp/gcc_make.html
- CMake: https://cmake.org/cmake/help/v3.31/guide/tutorial/index.html

CUDA:
- CUDA C++ Programming Guide. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA (Course) - Getting Started with Accelerated Computing in Modern CUDA C++
- NVIDIA (Course) - Fundamentals of Accelerated Computing with CUDA Python

PyTorch:
- https://docs.pytorch.org

PyTorch DDP:
- https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
- https://sebastianraschka.com/teaching/pytorch-1h/

Fundamental Deep Learning Books:
- Bishop, C. M., & Nasrabadi, N. M. (2006). Pattern recognition and machine learning (Vol. 4, No. 4, p. 738). New York: springer.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. The MIT press.
- Raschka, S., Liu, Y., Mirjalili, V., & Dzhulgakov, D. (2022). Machine learning with PyTorch and Scikit-Learn: Develop machine learning and deep learning models with Python. Packt.
- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). Dive into deep learning. Cambridge University Press.

Beginner level books:
- Rashid, T. (2016). Make Your own neural network. CreateSpace Independent Publishing Platform.
- Weidman, S. (2019). Deep learning from scratch: Building with Python from first principles (First edition). O’Reilly Media, Inc.
- Patterson, J., & Gibson, A. (2017). Deep learning: A practitioner’s approach (First edition). O’Reilly.

OpenCV:
- Kaehler, A., & Bradski, G. (2016). Learning OpenCV 3: computer vision in C++ with the OpenCV library. " O'Reilly Media, Inc.".
- Szeliski, R. (2022). Computer vision: algorithms and applications. Springer Nature.
- Прохоренок, Н. А. (2018). OpenCV и Java. Обработка изображений и компьютерное зрение. БХВ-Петербург.

YouTube:
- Евгений Разинков. (2025). AI: от основ до трансформеров. https://www.youtube.com/watch?v=MGmrCV6AbAI&list=PL6-BrcpR2C5Q1ivGTQcglILJG6odT2oCY
- Евгений Разинков. (2023). Machine Learning (2023, spring). https://www.youtube.com/playlist?list=PL6-BrcpR2C5SCyFvs9Xojv24povpBCI6W
- Евгений Разинков. (2022). Лекции по машинному обучению (осень, 2022). https://www.youtube.com/playlist?list=PL6-BrcpR2C5QYSAfoG8mbQUsI9zPVnlBV
- Евгений Разинков. (2021). Лекции по Advanced Computer Vision (2021). https://www.youtube.com/playlist?list=PL6-BrcpR2C5RV6xfpM7_k5321kJrcKEO0
- Евгений Разинков. (2021). Лекции по Deep Learning. https://www.youtube.com/playlist?list=PL6-BrcpR2C5QrLMaIOstSxZp4RfhveDSP
- Евгений Разинков. (2020). Лекции по компьютерному зрению. https://www.youtube.com/playlist?list=PL6-BrcpR2C5RZnmIWs6x0C2IZK6N9Z98I
- Евгений Разинков. (2019). Лекции по машинному обучению. https://www.youtube.com/playlist?list=PL6-BrcpR2C5RYoCAmC8VQp_rxSh0i_6C6
