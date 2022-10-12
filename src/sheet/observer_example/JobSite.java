package sheet.observer_example;

import java.util.ArrayList;
import java.util.List;

public class JobSite implements Observed{
    List<String> vacancies = new ArrayList<>();
    List<Observer> subscribers = new ArrayList<>();

    public void addVacancy(String v){
        this.vacancies.add(v);
        notifyObservers();
    }

    public void removeVacancy(String v){
        this.vacancies.remove(v);
        notifyObservers();
    }

    @Override
    public void addObserver(Observer ob) {
        this.subscribers.add(ob);
    }

    @Override
    public void removeObserver(Observer ob) {
        this.subscribers.remove(ob);
    }

    @Override
    public void notifyObservers() {
        for(Observer ob: subscribers){
            ob.handleEvent(this.vacancies);
        }
    }
}
