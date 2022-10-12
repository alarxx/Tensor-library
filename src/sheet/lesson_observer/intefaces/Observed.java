package sheet.lesson_observer.intefaces;

public interface Observed {

    void addObserver(Observer observer);
    void removeObserver(Observer observer);

    void applyToAll();
}
