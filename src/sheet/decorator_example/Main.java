package sheet.decorator_example;

public class Main {
    public static void main(String[] args) {
        Developer developer = new JavaDeveloper();
        developer = new SeniorDeveloper(developer);

        System.out.println(developer.makeJob());
    }
}
