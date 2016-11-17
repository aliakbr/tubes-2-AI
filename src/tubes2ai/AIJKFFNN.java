/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import weka.associations.NominalItem;
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
        hiddenLayerCount = Integer.parseInt(Utils.getOption("L", strings));
        hiddenLayerNeuronCount = Integer.parseInt(Utils.getOption("N", strings));
    }

    @Override
    public String[] getOptions() {
        return new String[]{"-L", String.valueOf(hiddenLayerCount), "-N", String.valueOf(hiddenLayerNeuronCount)};
    }

    private int hiddenLayerCount;
    private int hiddenLayerNeuronCount;
}
