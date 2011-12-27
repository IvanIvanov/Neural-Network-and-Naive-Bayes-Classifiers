package neuralnet;

import java.util.List;
import java.util.Random;

public class FeedforwardNeuralNetwork {
  
  private int n;
  private double[][] x;
  private double[][] delta;
  private double[][][] weight;
  private double[][][] previousWeightDelta;
  private Random random;

  public FeedforwardNeuralNetwork(int[] layers) {
    this(layers, 42);
  }

  public FeedforwardNeuralNetwork(int[] layers, int seed) {
    n = layers.length;
    x = new double[n][];
    delta = new double[n][];
    weight = new double[n][][];
    previousWeightDelta = new double[n][][];
    for(int i = 0; i < n; i++) {
      x[i] = new double[layers[i] + 1];
      delta[i] = new double[layers[i] + 1];
      if(i == 0) {
        weight[i] = new double[0][0];
        previousWeightDelta[i] = new double[0][0];
        continue;
      }
      weight[i] = new double[layers[i] + 1][layers[i - 1] + 1];
      previousWeightDelta[i] = new double[layers[i] + 1][layers[i - 1] + 1];
    }

    random = new Random(seed);
    for(int i = 0; i < weight.length; i++) {
      for(int j = 0; j < weight[i].length; j++) {
        for(int k = 0; k < weight[i][j].length; k++) {
          weight[i][j][k] = randomScalar();
        }
      }
    }
  }

  public void train(List <Example> examples,
                    int iterations,
                    double learningRate,
                    double momentum) {
    while(iterations-- > 0) {
      //System.out.println("#" + iterations + ": " + computeError(examples));
      for(Example example : examples) {
        forwardPropagate(example.getInput());
        backwardPropagateDelta(example.getOutput());
        updateWeights(learningRate, momentum);
      }
    }
  }

  public double[] computeOutput(double[] input) {
    return forwardPropagate(input);
  }

  public String toString() {
    StringBuilder result = new StringBuilder();
    String newLine = System.getProperty("line.separator");
    result.append("Neural network with " + n + " layers:" + newLine);
    for(int i = 0; i < n; i++) {
      result.append("  x:    ");
      for(int j = 0; j < x[i].length; j++) result.append(" " + x[i][j]);
      result.append(newLine);
      result.append("  delta:");
      for(int j = 0; j < delta[i].length; j++) result.append(" " + delta[i][j]);
      result.append(newLine);
      result.append("  weight:");
      result.append(newLine);
      result.append("  ");
      for(int j = 0; j < weight[i].length; j++) {
        result.append("[");
        for(int k = 0; k < weight[i][j].length; k++) {
          result.append(" " + weight[i][j][k]);
        }
        result.append(" ] ");
      }
      result.append(newLine);
    }
    return new String(result);
  }

  private double[] forwardPropagate(double[] input) {
    if(input == null || input.length + 1 != x[0].length) {
      throw new RuntimeException("Wrong number of input elements.");
    }

    x[0][x[0].length - 1] = 1.0;
    for(int i = 0; i < input.length; i++) {
      x[0][i] = input[i];
    }

    for(int i = 1; i < n; i++) {
      for(int j = 0; j < x[i].length; j++) {
        double sum = 0.0;
        for(int k = 0; k < weight[i][j].length; k++) {
          sum += weight[i][j][k] * x[i - 1][k];
        }
        x[i][j] = sigmoid(sum);
      }
      x[i][x[i].length - 1] = 1.0;
    }

    double[] output = new double[x[n - 1].length - 1];
    for(int i = 0; i < output.length; i++) {
      output[i] = x[n - 1][i];
    }

    return output;
  }

  private double backwardPropagateDelta(double[] output) {
    double error = 0.0;
    for(int i = 0; i < delta[n - 1].length - 1; i++) {
      double o = x[n - 1][i];
      delta[n - 1][i] = o * (1.0 - o) * (output[i] - o);
      error += Math.pow(output[i] - o, 2.0);
    }
    delta[n - 1][delta[n - 1].length - 1] = 0.0;

    for(int i = n - 2; i >= 0; i--) {
      for(int j = 0; j < delta[i].length; j++) {
        double sum = 0.0;
        for(int k = 0; k < delta[i + 1].length; k++) {
          sum += delta[i + 1][k] * weight[i + 1][k][j];
        }
        double o = x[i][j];
        delta[i][j] = o * (1.0 - o) * sum;
      }
    }

    return error;
  }

  private void updateWeights(double learningRate, double momentum) {
    for(int i = 1; i < n; i++) {
      for(int j = 0; j < weight[i].length; j++) {
        for(int k = 0; k < weight[i][j].length; k++) {
          double weightDelta = (learningRate * delta[i][j] * x[i - 1][k] +
              momentum * previousWeightDelta[i][j][k]);
          weight[i][j][k] += weightDelta;
          previousWeightDelta[i][j][k] = weightDelta;
        }
      }
    }
  }

  private double computeError(List <Example> examples) {
    double error = 0.0;
    for(Example example : examples) {
      forwardPropagate(example.getInput());
      double[] output = example.getOutput();
      for(int i = 0; i < x[n - 1].length - 1; i++) {
        error += Math.pow(output[i] - x[n - 1][i], 2.0);
      }
    }
    return error;
  }

  private double sigmoid(double x) {
    return 1.0 / (1.0 + Math.exp(-x));
  }

  private double randomScalar() {
    double num = random.nextDouble();
    return random.nextBoolean() ? -num : num;
  }
}

