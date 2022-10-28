package sheet.state_example;

public class Sleeping implements Activity{
    @Override
    public void justDoIt() {
        System.out.println("Sleeping (next day).");
    }
}
