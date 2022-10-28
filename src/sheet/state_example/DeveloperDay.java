package sheet.state_example;

public class DeveloperDay {
    public static void main(String[] args) {
        Developer developer = new Developer();

        Activity initialActivity = new Sleeping();
        developer.setActivity(initialActivity);

        for(int i=0; i<30; i++){
            developer.justDoIt();
            developer.changeActivity();
        }
    }
}
