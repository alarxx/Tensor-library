```
interface TensorInterface {
    int[] dims();

    Tensor set(Tensor item, int ... indexes);

    Tensor get(int ... indexes);

    Tensor fill(float item);

    int getLength();

    boolean isScalar();

    float getScalar();
    Tensor setScalar(float scalar);

    int rank();

    Tensor clone();
}  
  
interface AutoGradInterface {
    void _backward_();
    Tensor getGrad();
}  
  
interface OperationInterface {
    Tensor apply(Tensor src1, Tensor src2);
    Tensor apply(Tensor src);
}  


```
