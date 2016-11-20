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
    private final int MAX_ITERATIONS = 100;

    private Vector<Neuron> layerInput;
    private Vector<Neuron> hiddenLayer;
    private Vector<Neuron> layerOutput;
    List<Attribute> listAttribute;
    private int nHiddenLayer, nHiddenNeuron;
    private double learningRate;

    public AIJKFFNN() {
        nHiddenNeuron = 0;
        nHiddenLayer = 0;
        setSeed(0);
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);
        int nInputNeuron, nOutputNeuron;

        /* Inisialisasi tiap layer */
        nInputNeuron = instances.numAttributes()-1;
        nOutputNeuron = instances.numClasses();
        layerInput = new Vector<Neuron>(nInputNeuron);
        hiddenLayer = new Vector<Neuron>(nHiddenNeuron);
        layerOutput = new Vector<Neuron>(nOutputNeuron);
        Neuron input, hidden;

        Random random = new Random(getSeed());

        Enumeration<Attribute> attributeEnumeration = instances.enumerateAttributes();
        listAttribute = Collections.list(attributeEnumeration);

        /* Mengisi layer dengan neuron-neuron dengan weight default */
        for (int k = 0; k < nOutputNeuron; k++) {
            layerOutput.add(new Neuron());
        }

        for (int k = 0; k < nInputNeuron; k++) {
            layerInput.add(new Neuron());
        }

        /* Kalau ada hidden layer */
        if (nHiddenLayer > 0){
            for (int j = 0; j < nHiddenNeuron; j++) {
                hiddenLayer.add(new Neuron());
            }
        }

        /* Link */
        if (nHiddenLayer > 0) {
            linkNeurons(layerInput, hiddenLayer);
            linkNeurons(hiddenLayer, layerOutput);
        } else {
            linkNeurons(layerInput, layerOutput);
        }

        for (Neuron neuron : layerInput) {
            neuron.initialize(random);
        }

        if (nHiddenLayer > 0) {
            for (Neuron neuron : hiddenLayer) {
                neuron.initialize(random);
            }

        }

        for (Neuron neuron : layerOutput) {
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
                for (int ix = 0; ix < layerOutput.size(); ix++) {
                    if (ix == (int)instance.classValue()) {
                        layerOutput.get(ix).errorFromTarget(1);
                    }
                    else{
                        layerOutput.get(ix).errorFromTarget(0);
                    }
                }
                if (nHiddenLayer != 0){
                    for (Neuron nHid : hiddenLayer){
                        nHid.calculateError();
                    }
                }

            /* Update Weight */
                for (Neuron kHid : hiddenLayer){
                    kHid.updateWeights(learningRate);
                }
                for (Neuron kOut : layerOutput){
                    kOut.updateWeights(learningRate);
                }
            }

            iterations++;
        } while (iterations < MAX_ITERATIONS);


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
        double[] result = new double[layerOutput.size()];
        for (int i = 0; i < layerOutput.size(); i++) {
            result[i] = layerOutput.get(i).getOutput();
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

        return options.elements();
    }

    @Override
    public void setOptions(String[] strings) throws Exception {
        String hlc = Utils.getOption("L", strings);
        String hlnc = Utils.getOption("N", strings);
        if (hlc.length() > 0) {
            nHiddenLayer = Integer.parseInt(hlc);
        }

        if (hlnc.length() > 0) {
            nHiddenNeuron = Integer.parseInt(hlnc);
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
        for (int m = 0; m < layerInput.size(); m++) {
            double data = instance.value(listAttribute.get(m));
            layerInput.get(m).setOutput(data);
        }

            /* Menghitung output */
        if (nHiddenLayer != 0){
            for (Neuron eHid : hiddenLayer){
                eHid.calculateOutput();
            }
            for (Neuron eOut : layerOutput){
                eOut.calculateOutput();
            }
        }
        else{
            for (Neuron eOut : layerOutput){
                eOut.calculateOutput();
            }
        }
    }

    @Override
    public String[] getOptions() {
        return new String[]{"-L", String.valueOf(nHiddenLayer), "-N", String.valueOf(nHiddenNeuron)};
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

    private int seed;

    @Override
    public void setSeed(int i) {
        seed = i;
    }

    @Override
    public int getSeed() {
        return seed;
    }


}
