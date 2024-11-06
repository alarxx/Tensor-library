# Tensor-library
Analogue to Math and ML libraries like numpy or pytorch  
From scratch in plain java  

**[Dig deeper into the library](https://github.com/Alar-q/Tensor-library/tree/main/src/com/ml/lib)**

## Tensor 
Tensor - array of Tensors, except rank-0 Tensor (scalar)
```
[Tensor, Tensor, Tensor, Tensor]  
   |       |       |       |  
   |       |     [...]   [...]  
   |  [Tensor, Tensor, ...]  
[Tensor, Tensor...]  
```

![image](https://github.com/user-attachments/assets/225c35e3-ace0-4b2c-bc21-e461450093cd)  

---

## Features
* [Tensors](https://github.com/Alar-q/Tensor-library/tree/main/src/com/ml/lib/tensor)
* [Operations](https://github.com/Alar-q/Tensor-library/tree/main/src/com/ml/lib/core): transposition, matrix multiplication, convolution etc.
* Convenient creation of your own operations
* Dynamic computation graphs
* [AutoGrad](https://github.com/Alar-q/Tensor-library/tree/main/src/com/ml/lib/autograd)

## Usage Tips
* Build Machine and Deep Learning models  
* [Build Computer Vision algorithms](https://github.com/Alar-q/ML_library_JavaFX)
* Physical calculations

## Usage Examples
* [Linear Regression using Gradient Descent](https://github.com/Alar-q/Tensor-library/blob/main/src/com/ml/lib/nn/Main.java)  
* [Processing images loaded as a tensor](https://github.com/Alar-q/ML_library_JavaFX)

![image](https://github.com/user-attachments/assets/473d0569-8d2c-4a2d-9598-4d7ffc0ba8f7)  

## Limitations   
Element-by-element execution of operations, without GPU acceleration of calculations.  
I would like to add this feature, but learning OpenCL will take a lot of time. 
#### Explanation why I won't continue
"The problem with programmers   
is that when they make a car,  
they are at the same time, reinvent  
the wheel, steel mining methods 
and traffic rules"  
###### But maybe someday...

## References:
- Weidman, S. (2019). Deep learning from scratch: Building with Python from first principles (First edition). O’Reilly Media, Inc.
- Patterson, J., & Gibson, A. (2017). Deep learning: A practitioner’s approach (First edition). O’Reilly.
- Евгений Разинков. (2021). Лекции по Deep Learning. https://www.youtube.com/playlist?list=PL6-BrcpR2C5QrLMaIOstSxZp4RfhveDSP
- Raschka, S., Liu, Y., Mirjalili, V., & Dzhulgakov, D. (2022). Machine learning with PyTorch and Scikit-Learn: Develop machine learning and deep learning models with Python. Packt.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. The MIT press.
- Rashid, T. (2016). Make Your own neural network. CreateSpace Independent Publishing Platform.

## Authors

* **Alar Akilbekov** - [alarxx](https://github.com/alarxx) - [@alarxx](https://t.me/alarxx)

## Licence 
[MIT License](https://github.com/Alar-q/Tensor-library/blob/main/LICENSE)
