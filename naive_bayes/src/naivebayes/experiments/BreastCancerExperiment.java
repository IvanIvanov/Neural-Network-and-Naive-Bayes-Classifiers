package naivebayes.experiments;

import java.io.FileInputStream;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import naivebayes.CategoricalAttribute;
import naivebayes.ContinuousAttribute;
import naivebayes.Example;
import naivebayes.NaiveBayesClassifier;

public class BreastCancerExperiment {

  private static final String DEFAULT_DATA_FILE = "wdbc.data";
  private static final int DEFAULT_TRAINING_EXAMPLES = 300;

  private static final int INPUTS = 30;

  public static void main(String[] args) {
    String dataFile = DEFAULT_DATA_FILE;
    int trainingExamples = DEFAULT_TRAINING_EXAMPLES;
    for(int i = 0; i + 1 < args.length; i += 2) {
      if(args[i].equals("-f")) {
        dataFile = args[i + 1];
      }
      else if(args[i].equals("-e")) {
        trainingExamples = Integer.parseInt(args[i + 1]);
      }
    }

    List <Example> all = readFile(dataFile);

    Collections.shuffle(all, new Random(42));

    normalize(all);

    List <Example> examples = new ArrayList <Example> ();
    List <Example> validationSet = new ArrayList <Example> ();
    for(int i = 0; i < trainingExamples; i++) {
      examples.add(all.get(i));
    }
    for(int i = DEFAULT_TRAINING_EXAMPLES; i < all.size(); i++) {
      validationSet.add(all.get(i));
    }
    System.out.println(testClassifier(examples, validationSet));
  }

  private static double testClassifier(List <Example> examples,
                                     List <Example> validationSet) {

    NaiveBayesClassifier classifier = new NaiveBayesClassifier(examples);

    int correct = 0;
    for(Example example : validationSet) {
      String output = classifier.classify(
          example.getCategoricalInputs(),
          example.getContinuousInputs()).getValue();
      if(output.equals(example.getOutput().getValue())) {
        correct++;
      }
    }

    return (double)correct / validationSet.size();
  }

  private static List <Example> readFile(String filename) {
    List <Example> examples = new ArrayList <Example> ();
    try {
      Scanner sc = new Scanner(new FileInputStream(filename));
      while(sc.hasNextLine()) {
        examples.add(parseExample(sc.nextLine()));
      }
    }
    catch(IOException e) {
      e.printStackTrace();
    }
    return examples;
  }

  private static Example parseExample(String record) {
    String[] tokens = record.split(",");

    List <CategoricalAttribute> catInputs =
        new ArrayList <CategoricalAttribute> ();
    List <ContinuousAttribute> conInputs =
        new ArrayList <ContinuousAttribute> ();

    for(int i = 2; i < tokens.length; i++) {
      conInputs.add(new ContinuousAttribute(
          Double.parseDouble(tokens[i])));
    }

    return new Example(
        catInputs,
        conInputs,
        new CategoricalAttribute(tokens[1]));
  }

  private static void normalize(List <Example> examples) {
    for(int i = 0; i < INPUTS; i++) {
      double minValue =
          examples.get(0).getContinuousInputs().get(i).getValue();
      double maxValue =
          examples.get(0).getContinuousInputs().get(i).getValue();
      for(int j = 0; j < examples.size(); j++) {
        minValue = Math.min(
            minValue,
            examples.get(j).getContinuousInputs().get(i).getValue());
        maxValue = Math.max(
            maxValue,
            examples.get(j).getContinuousInputs().get(i).getValue());
      }
      for(int j = 0; j < examples.size(); j++) {
        double value =
            examples.get(j).getContinuousInputs().get(i).getValue();
        value = (value - minValue) / (maxValue - minValue);
        examples.get(j).getContinuousInputs().set(
            i, new ContinuousAttribute(value));
      }
    }
  }
}

