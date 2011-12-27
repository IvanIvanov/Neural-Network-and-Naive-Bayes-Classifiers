package naivebayes;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class NaiveBayesClassifier {

  private int nCategoricalInputs;
  private int nContinuousInputs;
  private List <String> classes;
  private Map <String, Double> classProbabilityTable;
  private Map <String, List <Map <String, Double> > > categoricalCPT;
  private Map <String, List <Pair <Double, Double> > > continuousCPT;

  public NaiveBayesClassifier(List <Example> examples) {
    train(examples);
  }

  public CategoricalAttribute classify(
      List <CategoricalAttribute> categoricalInputs,
      List <ContinuousAttribute> continuousInputs) {
    String bestClass = classes.get(0);
    double bestClassProbability = 0.0;
    for(String className : classes) {
      double classProbability = computeClassProbability(
          className, categoricalInputs, continuousInputs);
      if(classProbability > bestClassProbability) {
        bestClassProbability = classProbability;
        bestClass = className;
      }
    }
    return new CategoricalAttribute(bestClass);
  }

  private double computeClassProbability(
      String className,
      List <CategoricalAttribute> categoricalInputs,
      List <ContinuousAttribute> continuousInputs) {
    double prob = classProbabilityTable.get(className);

    int index = 0;
    for(Map <String, Double> distribution : categoricalCPT.get(className)) {
      if(distribution.containsKey(categoricalInputs.get(index).getValue())) {
        prob *= distribution.get(categoricalInputs.get(index).getValue());
      }
      else {
        prob *= 0.0;
      }
      index++;
    }

    index = 0;
    for(Pair <Double, Double> distribution : continuousCPT.get(className)) {
      prob *= gaussianValue(
          distribution.getFirst(),
          distribution.getSecond(),
          continuousInputs.get(index).getValue());
      index++;
    }

    return prob;
  }

  private void train(List <Example> examples) {
    classes = uniqueOutputValues(examples);
    classProbabilityTable = new HashMap <String, Double> ();
    categoricalCPT =
        new HashMap <String, List <Map <String, Double> > > ();
    continuousCPT =
        new HashMap <String, List <Pair <Double, Double> > > ();

    int n = examples.size();
    if(n == 0) return;
    nContinuousInputs = examples.get(0).getContinuousInputs().size();
    nCategoricalInputs = examples.get(0).getCategoricalInputs().size();

    List <List <String> > uniqueValues = new ArrayList <List <String> > ();
    for(int i = 0; i < nCategoricalInputs; i++) {
      uniqueValues.add(uniqueCategoricalInputValues(examples, i));
    }

    for(String output : uniqueOutputValues(examples)) {
      List <Example> filteredExamples = filterExamplesByOutput(
          examples, output);
      classProbabilityTable.put(
          output,
          (double) filteredExamples.size() / n);
      categoricalCPT.put(output, new ArrayList <Map <String, Double> > ());
      continuousCPT.put(output, new ArrayList <Pair <Double, Double> > ());
      for(int i = 0; i < nCategoricalInputs; i++) {
        categoricalCPT.get(output).add(
            categoryFrequencyMap(
                filteredExamples, i, uniqueValues.get(i)));
      }
      for(int i = 0; i < nContinuousInputs; i++) {
        continuousCPT.get(output).add(
            normalDistributionOfInput(filteredExamples, i));
      }
    }
  }

  private List <String> uniqueOutputValues(List <Example> examples) {
    Set <String> seen = new HashSet <String> ();
    for(Example example : examples) {
      if(!seen.contains(example.getOutput().getValue())) {
        seen.add(example.getOutput().getValue());
      }
    }
    return new ArrayList <String> (seen);
  }

  private List <String> uniqueCategoricalInputValues(
      List <Example> examples,
      int categoryIndex) {
    Set <String> seen = new HashSet <String> ();
    for(Example example : examples) {
      String value =
          example.getCategoricalInputs().get(categoryIndex).getValue();
      if(!seen.contains(value)) {
        seen.add(value);
      }
    }
    return new ArrayList <String> (seen);
  }

  private List <Example> filterExamplesByOutput(
      List <Example> examples,
      String value) {
    List <Example> result = new ArrayList <Example> ();
    for(Example example : examples) {
      if(example.getOutput().getValue().equals(value)) {
        result.add(example);
      }
    }
    return result;
  }

  private Map <String, Double> categoryFrequencyMap(
      List <Example> examples,
      int categoryIndex,
      List <String> uniqueValues) {
    int n = examples.size();
    int m = uniqueValues.size();
    Map <String, Double> freq = new HashMap <String, Double> ();
    for(String value : uniqueValues) {
      freq.put(value, 1.0 / (n + m));
    }
    for(Example example : examples) {
      String value = example.getCategoricalInputs().get(
          categoryIndex).getValue();
      freq.put(value, freq.get(value) + 1.0 / (n + m));
    }
    return freq;
  }

  private Pair <Double, Double> normalDistributionOfInput(
      List <Example> examples,
      int index) {
    List <Double> values = new ArrayList <Double> ();
    for(Example example : examples) {
      values.add(example.getContinuousInputs().get(index).getValue());
    }

    double mean = computeMean(values);
    double variance = computeVariance(values, mean);

    return new Pair <Double, Double> (mean, variance);
  }

  private double computeMean(List <Double> values) {
    int n = values.size();
    double sum = 0.0;
    for(int i = 0; i < n; i++) sum += values.get(i);
    return (n == 0 ? 0.0 : sum / n);
  }

  private double computeVariance(List <Double> values, double mean) {
    int n = values.size();
    double sum = 0.0;
    for(int i = 0; i < n; i++) {
      double diff = values.get(i) - mean;
      sum += diff * diff;
    }
    return (sum < 1e-8 ? 1.0 : sum / n);
  }

  private double gaussianValue(double mean, double variance, double x) {
    return 1.0 / Math.sqrt(2.0 * Math.PI * variance) *
        Math.exp(-((x - mean) * (x - mean)) / (2.0 * variance));
  }
}

