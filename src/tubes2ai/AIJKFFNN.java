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

import java.io.Serializable;
import java.util.*;

/**
 *
 * @author i
 */
public class AIJKFFNN implements Classifier, OptionHandler, CapabilitiesHandler, Serializable {

    public AIJKFFNN() {
        hiddenLayerCount = 0;
        hiddenLayerNeuronCount = 0;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);
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
