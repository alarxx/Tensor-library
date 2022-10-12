package sheet.observer_example;

public class Main {
    public static void main(String[] args) {
        JobSite site = new JobSite();

        site.addVacancy("Java");
        site.addVacancy("JavaScript");

        Observer s1 = new Subscriber("Alar");
        Observer s2 = new Subscriber("Tomiris");

        site.addObserver(s1);
        site.addObserver(s2);

        site.addVacancy("Python");
        site.addVacancy("C++");

        site.removeVacancy("Java");
    }
}
