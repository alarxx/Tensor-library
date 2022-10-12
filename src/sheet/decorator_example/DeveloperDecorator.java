package sheet.decorator_example;

public class DeveloperDecorator implements Developer {
    protected Developer developer;

    public DeveloperDecorator(Developer developer){
        this.developer = developer;
    }

    @Override
    public String makeJob(){
        return developer.makeJob();
    }
}
