/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import weka.classifiers.CheckClassifier;
import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.Capabilities.Capability;

import java.util.Enumeration;
import java.lang.*;
import java.util.*;
import java.io.Serializable;

/**
 *
 * @author i
 */
public class AIJKFFNN implements Classifier, OptionHandler, CapabilitiesHandler, Serializable, Randomizable {
    private Vector<Neuron> inputLayer;
    private Vector<Neuron> hiddenLayer;
    private Vector<Neuron> outputLayer;

    private Neuron[] inputLayerArray;
    private Neuron[] outputCalculationArray;

    private List<Attribute> attributeList;
    private int nHiddenLayer, nHiddenNeuron;
    private double learningRate;
    private int seed;
    private int maxIterations;

    public AIJKFFNN() {
        nHiddenNeuron = 0;
        nHiddenLayer = 0;
        maxIterations = 100;
        learningRate = 1;
        setSeed(1);
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);
        int nInputNeuron, nOutputNeuron;

        /* Inisialisasi tiap layer */
        nInputNeuron = instances.numAttributes()-1;
        nOutputNeuron = instances.numClasses();
        inputLayer = new Vector<Neuron>(nInputNeuron);
        hiddenLayer = new Vector<Neuron>(nHiddenNeuron);
        outputLayer = new Vector<Neuron>(nOutputNeuron);

        Random random = new Random(getSeed());

        Enumeration<Attribute> attributeEnumeration = instances.enumerateAttributes();
        attributeList = Collections.list(attributeEnumeration);

        /* Mengisi layer dengan neuron-neuron dengan weight default */
        for (int k = 0; k < nOutputNeuron; k++) {
            outputLayer.add(new Neuron());
        }

        for (int k = 0; k < nInputNeuron; k++) {
            inputLayer.add(new Neuron());
        }

        /* Kalau ada hidden layer */
        if (nHiddenLayer > 0){
            for (int j = 0; j < nHiddenNeuron; j++) {
                hiddenLayer.add(new Neuron());
            }
        }

        /* Link */
        if (nHiddenLayer > 0) {
            linkNeurons(inputLayer, hiddenLayer);
            linkNeurons(hiddenLayer, outputLayer);
        } else {
            linkNeurons(inputLayer, outputLayer);
        }

        for (Neuron neuron : inputLayer) {
            neuron.initialize(random);
        }

        inputLayerArray = new Neuron[nInputNeuron];
        int i = 0;
        for (Neuron neuron : inputLayer) {
            inputLayerArray[i] = neuron;
            i++;
        }

        outputCalculationArray = new Neuron[nHiddenLayer*nHiddenNeuron + nOutputNeuron];
        int j = 0;
        for (Neuron neuron : hiddenLayer) {
            outputCalculationArray[j] = neuron;
            j++;
        }
        for (Neuron neuron : outputLayer) {
            outputCalculationArray[j] = neuron;
            j++;
        }


        if (nHiddenLayer > 0) {
            for (Neuron neuron : hiddenLayer) {
                neuron.initialize(random);
            }

        }

        for (Neuron neuron : outputLayer) {
            neuron.initialize(random);
        }

        /* Learning */
        int iterations = 0;
        List<Double> errors = new ArrayList<>();
        do {
            for (Instance instance : instances) {
            /* Memasukkan instance ke input neuron */
                loadInput(instance);

            /* Menghitung error dari layer output ke input */
            /* Menyiapkan nilai target */
                for (int ix = 0; ix < outputLayer.size(); ix++) {
                    if (ix == (int)instance.classValue()) {
                        outputLayer.get(ix).errorFromTarget(1);
                    }
                    else{
                        outputLayer.get(ix).errorFromTarget(0);
                    }
                }
                if (nHiddenLayer != 0){
                    for (Neuron nHid : hiddenLayer){
                        nHid.calculateError();
                    }
                }

            /* Update Weight */

                for (int k = 0; k < outputCalculationArray.length; k++) {
                    outputCalculationArray[k].updateWeights(learningRate);
                }
            }

            iterations++;

            if (iterations % 500 == 0) {
                System.out.println("FFNN iteration " + iterations);
            }

        } while (iterations < maxIterations);


    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] dist = distributionForInstance(instance);
        int max = 0;
        for (int i = 1; i < dist.length; i++) {
            if (dist[max] < dist[i]) {
                max = i;
            }
        }
        return max;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        loadInput(instance);
        double[] result = new double[outputLayer.size()];
        for (int i = 0; i < outputLayer.size(); i++) {
            result[i] = outputLayer.get(i).getOutput();
        }
        return result;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities c = new Capabilities(this);
        c.enable(Capability.NUMERIC_ATTRIBUTES);
        c.enable(Capability.NOMINAL_CLASS);
        return c;
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> options = new Vector<>();

        options.add(new Option("Amount of hidden layers", "L", 1, "-L <amount>"));
        options.add(new Option("Amount of neurons in hidden layer", "N", 1, "-N <amount>"));
        options.add(new Option("Number of iterations", "I", 1, "-I <amount>"));
        options.add(new Option("Learning rate", "R", 1, "-R <amount>"));

        return options.elements();
    }

    @Override
    public void setOptions(String[] strings) throws Exception {
        String hlc = Utils.getOption("L", strings);
        String hlnc = Utils.getOption("N", strings);
        String it = Utils.getOption("I", strings);
        String lr = Utils.getOption("R", strings);
        if (hlc.length() > 0) {
            nHiddenLayer = Integer.parseInt(hlc);
        }

        if (hlnc.length() > 0) {
            nHiddenNeuron = Integer.parseInt(hlnc);
        }

        if (it.length() > 0) {
            maxIterations = Integer.parseInt(it);
        }

        if (lr.length() > 0) {
            learningRate = Double.parseDouble(lr);
        }

    }

    private void linkNeurons(List<Neuron> from, List<Neuron> to) {
        for (Neuron neuronSrc : from) {
            for (Neuron neuronDest : to) {
                neuronSrc.linkTo(neuronDest);
            }
        }
    }

    private void loadInput(Instance instance) {
        for (int m = 0; m < inputLayer.size(); m++) {
            double data = instance.value(attributeList.get(m));
            inputLayerArray[m].setOutput(data);
        }

            /* Menghitung output */
        for (int i = 0; i < outputCalculationArray.length; i++) {
            outputCalculationArray[i].calculateOutput();
        }
    }

    @Override
    public String[] getOptions() {
        return new String[]{
            "-L", String.valueOf(nHiddenLayer),
            "-R", String.valueOf(learningRate),
            "-N", String.valueOf(nHiddenNeuron),
            "-I", String.valueOf(maxIterations)
        };
    }

    public static void main(String[] args) {
        CheckClassifier checker = new CheckClassifier();
        try {
            checker.setOptions(Utils.splitOptions("-W AIJKFFNN"));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        checker.doTests();

        CheckOptionHandler optionChecker = new CheckOptionHandler();
        try {
            optionChecker.setOptions(Utils.splitOptions("-W AIJKFFNN"));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        optionChecker.doTests();
    }

    @Override
    public void setSeed(int i) {
        seed = i;
    }

    @Override
    public int getSeed() {
        return seed;
    }


}
