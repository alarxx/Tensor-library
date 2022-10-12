package sheet.observer_example;

public interface Observed {
    void addObserver(Observer ob);

    void removeObserver(Observer ob);

    void notifyObservers();
}
