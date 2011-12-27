package naivebayes;

import java.util.List;

public class Example {

  private List <CategoricalAttribute> categoricalInputs;
  private List <ContinuousAttribute> continuousInputs;
  private CategoricalAttribute output;

  public Example(
      List <CategoricalAttribute> categoricalInputs,
      List <ContinuousAttribute> continuousInputs,
      CategoricalAttribute output) {
    this.categoricalInputs = categoricalInputs;
    this.continuousInputs = continuousInputs;
    this.output = output;
  }

  public List <CategoricalAttribute> getCategoricalInputs() {
    return categoricalInputs;
  }

  public List <ContinuousAttribute> getContinuousInputs() {
    return continuousInputs;
  }

  public CategoricalAttribute getOutput() {
    return output;
  }

  public String toString() {
    StringBuilder result = new StringBuilder();
    result.append("Example: " + System.getProperty("line.separator"));
    
    result.append("Categorical Attributes:");
    for(CategoricalAttribute attribute : categoricalInputs) {
      result.append(" " + attribute.getValue());
    }
    result.append(System.getProperty("line.separator"));

    result.append("Continuous Attributes:");
    for(ContinuousAttribute attribute : continuousInputs) {
      result.append(" " + attribute.getValue());
    }
    result.append(System.getProperty("line.separator"));

    return result.toString();
  }
}

