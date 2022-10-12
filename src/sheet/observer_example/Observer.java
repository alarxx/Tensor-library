package sheet.observer_example;

import java.util.List;

public interface Observer {
    void handleEvent(List<String> vacancies);
}
