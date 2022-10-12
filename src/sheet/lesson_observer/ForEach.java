package sheet.lesson_observer;


import sheet.lesson_observer.intefaces.Observed;
import sheet.lesson_observer.intefaces.Observer;
import sheet.lesson_observer.intefaces.Operation;

import java.util.LinkedList;
import java.util.List;

public class ForEach implements Observed {
    private List<Piece> pieces;
    private Operation operation;

    public ForEach(){
        pieces = new LinkedList<>();
    }

    @Override
    public void addObserver(Observer observer) {
        pieces.add((Piece) observer);
    }

    @Override
    public void removeObserver(Observer observer) {
        pieces.remove((Piece) observer);
    }

    public void apply(Operation operation){
        this.operation = operation;
        applyToAll();
    }

    @Override
    public void applyToAll() {
        for(Piece p: pieces){
            p.apply(operation);
        }
    }

}
