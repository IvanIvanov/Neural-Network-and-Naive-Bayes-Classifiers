package naivebayes.experiments;

import java.io.FileInputStream;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

import naivebayes.CategoricalAttribute;
import naivebayes.ContinuousAttribute;
import naivebayes.Example;
import naivebayes.NaiveBayesClassifier;

public class AbaloneExperiment {

  private static final String DEFAULT_DATA_FILE = "abalone.data";
  private static final int DEFAULT_TRAINING_EXAMPLES = 3133;

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
      if(getClass(output) == getClass(example.getOutput().getValue())) {
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

    catInputs.add(new CategoricalAttribute(tokens[0]));
    for(int i = 1; i < tokens.length - 1; i++) {
      conInputs.add(new ContinuousAttribute(
          Double.parseDouble(tokens[i])));
    }

    return new Example(
        catInputs,
        conInputs,
        new CategoricalAttribute(tokens[tokens.length - 1]));
  }

  private static int getClass(String s) {
    int num = Integer.parseInt(s);
    if(num <= 8) return 1;
    if(num <= 10) return 2;
    return 3;
  }
}

