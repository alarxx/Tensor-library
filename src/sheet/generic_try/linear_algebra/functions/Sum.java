package sheet.generic_try.linear_algebra.functions;

import sheet.generic_try.linear_algebra.Tensor;
import sheet.generic_try.linear_algebra.interfaces.Function;

public class Sum implements Function {
    @Override
    public Tensor function(Tensor t1, Tensor t2) {
        if(t1.dimsEqual(t2)){
            if(t1.dims().length < 2){
                System.out.println("Vector");
            }

            int rows = t1.dims()[0];
            int cols = t2.dims()[1];

            Tensor result = new Tensor(t1.dims());
            System.out.println(t1.size());
            for(int i=0; i<t1.size(); i++){
                result.__array()[i] = (Float) t1.__array()[i] + (Float) t2.__array()[i];
            }
            return result;
        }
        return null;
    }
}
