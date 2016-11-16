/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import weka.classifiers.Classifier;
import weka.core.*;

import java.util.Enumeration;

/**
 *
 * @author i
 */
public class AIJKFFNN implements Classifier, OptionHandler {

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
        return null;
    }

    @Override
    public Enumeration<Option> listOptions() {
        return null;
    }

    @Override
    public void setOptions(String[] strings) throws Exception {

    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}
