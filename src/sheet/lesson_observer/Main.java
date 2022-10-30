package sheet.lesson_observer;

import com.ml.lib.tensor.Tensor;

import sheet.lesson_observer.operations.Mult;
import sheet.lesson_observer.operations.Plus;

import java.util.Scanner;

import static com.ml.lib.core.Core.allTensorOfRank;
import static com.ml.lib.tensor.Tensor.tensor;

/**
 *
 * Tensor of 2nd rank - vector
 * Каждый скаляр вектора считается за обсервер Piece, а ForEach Observed
 *
 * Мы меняем состояние(field) в ForEach и он передает, уведомляет каждый скаляр об изменении,
 * в то время как скаляр обрабатывает это изменение.
 *
 * */
public class Main {
    public static void main(String[] args) {
        Tensor tensor = new Tensor(3);

        tensor.set(tensor(0), 0);
        tensor.set(tensor(2), 1);
        tensor.set(tensor(5), 2);

        System.out.println(tensor);

        ForEach forEach = new ForEach();

        for(Tensor t: allTensorOfRank(tensor, 0)){
            forEach.addObserver(new Piece(t));
        }

        Scanner scanner = new Scanner(System.in);
        while(true){
            System.out.print("\noperation: ");
            String message = scanner.next();

            System.out.print("\nvalue: ");
            float v = scanner.nextFloat();

            if(message.equals("+")){
                forEach.apply(new Plus(v));
            }
            else if(message.equals("*")){
                forEach.apply(new Mult(v));
            }
            else if(message.equals("-")){
                forEach.apply(new Plus(v));
            }
            else if(message.equals("/")){
                forEach.apply(new Mult(v));
            }
            else{
                System.out.println("something wrong");
            }
        }
//        System.out.println("First Operation");
//        forEach.apply(new Plus(1));
//
//        System.out.println("Second Operation");
//        forEach.apply(new Mult(2));


    }

}
