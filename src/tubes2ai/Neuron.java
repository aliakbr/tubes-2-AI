package tubes2ai;

import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

class Neuron {
    public Neuron() {
        inputs = new HashMap<>();
        outputs = new Vector<>();
        bias = 1;
    }

    void calculateOutput() {
        double x = bias;
        for (Map.Entry<Neuron, Double> entry : inputs.entrySet()) {
            x += entry.getKey().getOutput() * entry.getValue();
        }

        outputValue = sigmoid(x);
    }

    void updateWeights(double learningRate) {
        for (Neuron neuron : inputs.keySet()) {
            double prevWeight = this.getWeight(neuron);
            double newWeight = prevWeight + error * neuron.getOutput();

            inputs.put(neuron, newWeight);
        }
    }

    void calculateError() {
        double errsum = 0;
        for (Neuron output : outputs) {
            errsum += output.getWeight(this) * output.getError();
        }

        error = errsum * (1 - outputValue) * outputValue;

    }

    public double getError() {
        return error;
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
        return inputs.get(n);
    }

    void linkTo(Neuron outDest) {
        outputs.add(outDest);
        outDest.inputs.put(this, 1.0);
    }

    private double error;
    private double outputValue;
    private double bias;
    private Map<Neuron, Double> inputs;
    private Vector<Neuron> outputs;

    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}
