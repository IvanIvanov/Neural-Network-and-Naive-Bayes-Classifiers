package neuralnet.experiments;

import java.io.FileInputStream;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

import neuralnet.Example;
import neuralnet.FeedforwardNeuralNetwork;

public class BreastCancerExperiment {

  private static final String DEFAULT_DATA_FILE = "wdbc.data";
  private static final int DEFAULT_TRAINING_EXAMPLES = 300;
  private static final int DEFAULT_HIDDEN_LAYERS = 1;
  private static final int DEFAULT_HIDDEN_LAYER_SIZE = 4;
  private static final int DEFAULT_ITERATIONS = 1000;
  private static final double DEFAULT_LEARNING_RATE = 0.1;
  private static final double DEFAULT_MOMENTUM = 0.1;

  private static final int INPUTS = 30;
  private static final int OUTPUTS = 2;

  public static void main(String[] args) {
    String dataFile = DEFAULT_DATA_FILE;
    int trainingExamples = DEFAULT_TRAINING_EXAMPLES;
    int hiddenLayers = DEFAULT_HIDDEN_LAYERS;
    int iterations = DEFAULT_ITERATIONS;
    double learningRate = DEFAULT_LEARNING_RATE;
    double momentum = DEFAULT_MOMENTUM;
    Map <Integer, Integer> layerSizes = new HashMap <Integer, Integer> ();
    for(int i = 0; i + 1 < args.length; i += 2) {
      if(args[i].equals("-f")) {
        dataFile = args[i + 1];
      }
      else if(args[i].equals("-h")) {
        hiddenLayers = Integer.parseInt(args[i + 1]);
      }
      else if(args[i].startsWith("-n")) {
        layerSizes.put(
            Integer.parseInt(args[i].substring(2)),
            Integer.parseInt(args[i + 1]));
      }
      else if(args[i].equals("-r")) {
        learningRate = Double.parseDouble(args[i + 1]);
      }
      else if(args[i].equals("-m")) {
        momentum = Double.parseDouble(args[i + 1]);
      }
      else if(args[i].equals("-e")) {
        trainingExamples = Integer.parseInt(args[i + 1]);
      }
      else if(args[i].equals("-i")) {
        iterations = Integer.parseInt(args[i + 1]);
      }
    }

    int[] layers = new int[2 + hiddenLayers];
    layers[0] = INPUTS;
    layers[layers.length - 1] = OUTPUTS;
    for(int i = 1; i <= hiddenLayers; i++) {
      if(layerSizes.containsKey(i)) {
        layers[i] = layerSizes.get(i);
      }
      else {
        layers[i] = DEFAULT_HIDDEN_LAYER_SIZE;
      }
    }

    List <Example> all = readFile(dataFile);
    List <Example> examples = new ArrayList <Example> ();
    List <Example> validationSet = new ArrayList <Example> ();

    normalize(all);

    Collections.shuffle(all, new Random(42));
    for(int i = 0; i < trainingExamples; i++) {
      examples.add(all.get(i));
    }
    for(int i = DEFAULT_TRAINING_EXAMPLES; i < all.size(); i++) {
      validationSet.add(all.get(i));
    }

    double percent =
        testClassifier(
            layers,
            examples,
            iterations,
            learningRate,
            momentum,
            validationSet);
    
    System.out.println(percent); 
  }

  private static double testClassifier(int[] layers,
                                       List <Example> examples,
                                       int iterations,
                                       double learningRate,
                                       double momentum,
                                       List <Example> validationSet) {

    FeedforwardNeuralNetwork net = new FeedforwardNeuralNetwork(layers);
    net.train(examples, iterations, learningRate, momentum);

    int correct = 0;
    for(Example example : validationSet) {
      double[] output = net.computeOutput(example.getInput());
      if(getMaxIndex(output) == getMaxIndex(example.getOutput())) correct++;
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
    double[] input = new double[INPUTS];
    for(int i = 2; i < tokens.length; i++) {
      input[i - 2] = Double.parseDouble(tokens[i]);
    }

    double[] output = new double[OUTPUTS];
    output[tokens[1].equals("B") ? 0 : 1] = 1.0;

    return new Example(input, output);
  }

  private static int getMaxIndex(double[] x) {
    int index = 0;
    for(int i = 0; i < x.length; i++) {
      if(x[i] > x[index]) index = i;
    }
    return index;
  }

  private static void normalize(List <Example> examples) {
    for(int i = 0; i < INPUTS; i++) {
      double minValue = examples.get(0).getInput()[i];
      double maxValue = examples.get(0).getInput()[i];
      for(int j = 0; j < examples.size(); j++) {
        minValue = Math.min(minValue, examples.get(j).getInput()[i]);
        maxValue = Math.max(maxValue, examples.get(j).getInput()[i]);
      }
      for(int j = 0; j < examples.size(); j++) {
        double value = examples.get(j).getInput()[i];
        value = (value - minValue) / (maxValue - minValue);
        examples.get(j).getInput()[i] = value;
      }
    }
  }
}

