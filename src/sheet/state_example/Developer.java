package sheet.state_example;

public class Developer {
    private Activity activity;

    public void setActivity(Activity activity) {
        this.activity = activity;
    }

    public void justDoIt(){
        activity.justDoIt();
    }

    public void changeActivity(){
        if(activity instanceof Sleeping){
            setActivity(new Eating());
        }
        else if(activity instanceof Eating){
            setActivity(new Reading());
        }
        else if(activity instanceof Reading){
            setActivity(new Coding());
        }
        else if(activity instanceof Coding){
            setActivity(new Training());
        }
        else if(activity instanceof Training){
            setActivity(new Sleeping());
        }
    }
}
