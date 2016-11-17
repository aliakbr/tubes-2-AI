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
public class AIJKFFNN implements Classifier, OptionHandler, CapabilitiesHandler {
    private Vector<Neuron> layerInput;
    private Vector<Neuron> hiddenLayer;
    private Vector<Neuron> layerOutput;
    private int nHiddenLayer, nHiddenNeuron;
    private double learningRate;

    public AIJKFFNN() {
        hiddenLayerCount = 0;
        hiddenLayerNeuronCount = 0;
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

        Enumeration<Attribute> attributeEnumeration = instances.enumerateAttributes();
        List<Attribute> listAttribute = Collections.list(attributeEnumeration);

        /* Mengisi layer dengan neuron-neuron dengan weight default */
        for (int k = 0; k < nOutputNeuron; k++) {
            layerOutput.add(new Neuron());
        }

        /* Kalau ada hidden layer */
        if (nHiddenLayer != 0){
            for (int j = 0; j < nHiddenNeuron; j++) {
                hiddenLayer.add(new Neuron());
                /* Link ke output */
                for (Neuron e : layerOutput){
                    hiddenLayer.get(j).linkTo(e);
                }
            }

            for (int k = 0; k < nInputNeuron; k++) {
                layerInput.add(new Neuron());
                /* Link ke hidden */
                for (Neuron eF : hiddenLayer){
                    layerInput.get(k).linkTo(eF);
                }
            }
        }
        /* Jika tidak ada hidden layer */
        else{
            for (int l = 0; l < nInputNeuron; l++) {
                layerInput.add(new Neuron());
                for (Neuron eL : layerOutput){
                    layerInput.get(l).linkTo(eL);
                }
            }
        }

        /* Learning */
        double data;
        for (Instance instance : instances) {
            /* Memasukkan instance ke input neuron */
            for (int m = 0; m < layerInput.size(); m++) {
                data = instance.value(listAttribute.get(m));
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

    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
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
            hiddenLayerCount = Integer.parseInt(hlc);
        }

        if (hlnc.length() > 0) {
            hiddenLayerNeuronCount = Integer.parseInt(hlnc);
        }

    }

    @Override
    public String[] getOptions() {
        return new String[]{"-L", String.valueOf(hiddenLayerCount), "-N", String.valueOf(hiddenLayerNeuronCount)};
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

    private int hiddenLayerCount;
    private int hiddenLayerNeuronCount;
}
