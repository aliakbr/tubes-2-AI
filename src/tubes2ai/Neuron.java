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
        /* */
    }

    void calculateError() {

    }

    void setError(double e) {
        error = e;
    }

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
