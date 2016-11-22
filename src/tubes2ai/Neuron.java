package tubes2ai;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Vector;

class Neuron implements Serializable {
    public Neuron() {
        inputsV = new Vector<>();
        outputsV = new Vector<>();
        inputIndexes = new HashMap<>();
        bias = 1;
    }

    void calculateOutput() {
        double x = bias;
        for (int i = 0; i < inputNeurons.length; i++) {
            x += inputNeurons[i].getOutput() * inputWeights[i];
        }

        outputValue = sigmoid(x);
    }

    void updateWeights(double learningRate) {
        for (int i = 0; i < inputNeurons.length; i++) {
            double prevWeight = inputWeights[i];
            double newWeight = prevWeight + learningRate * error * inputNeurons[i].getOutput();
            inputWeights[i] = newWeight;
        }

        bias += error;
    }

    void calculateError() {
        double errsum = 0;
        for (int i = 0; i < outputNeurons.length; i++) {
            Neuron output = outputNeurons[i];
            errsum += output.getWeight(this) * output.getError();
        }

        error = errsum * (1 - outputValue) * outputValue;
    }

    public double getError() {
        return error;
    }

    public void initialize(Random random) {
        int inputCount = inputsV.size();
        inputNeurons = new Neuron[inputCount];
        inputWeights = new double[inputCount];

        outputNeurons = new Neuron[outputsV.size()];

        int i = 0;
        for (Neuron neuron : inputsV) {
            inputNeurons[i] = neuron;
            inputIndexes.put(neuron, i);
            i++;
        }
        int j = 0;
        for (Neuron neuron : outputsV) {
            outputNeurons[j] = neuron;
            j++;
        }


        reinitializeWeights(random);
    }

    private void reinitializeWeights(Random random) {
        int inputCount = inputNeurons.length;
        double initWeightLimit = 1 / Math.sqrt(inputCount);
        for (int i = 0; i < inputCount; i++) {
            inputWeights[i] = random.nextDouble()*initWeightLimit*2 - initWeightLimit;
        }
    }

    /* Hanya untuk output */
    void errorFromTarget(double target) {
        error = outputValue * (1 - outputValue) * (target - outputValue);
    }

    /* Hanya untuk input */
    void setOutput(double x) {
        outputValue = x;
    }

    double getOutput() {
        return outputValue;
    }

    double getWeight(Neuron n) {
        return inputWeights[inputIndexes.get(n)];
    }

    void linkTo(Neuron outDest) {
        outputsV.add(outDest);
        outDest.inputsV.add(this);
    }

    private double error;
    private double outputValue;
    private double bias;
    private Map<Neuron, Integer> inputIndexes;
    private Vector<Neuron> inputsV;
    private Vector<Neuron> outputsV;
    private Neuron[] inputNeurons;
    private Neuron[] outputNeurons;
    private double[] inputWeights;

    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}
