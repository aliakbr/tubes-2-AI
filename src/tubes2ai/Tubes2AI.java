/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

import java.util.Random;

/**
 *
 * @author i
 */
public class Tubes2AI {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        ConverterUtils.DataSource source;

        try {
            AIJKFFNN classifier = new AIJKFFNN();
            classifier.setOptions(Utils.splitOptions(""));
            source = new ConverterUtils.DataSource("data/iris.arff");
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            Evaluation eval10fold = new Evaluation(data);
            eval10fold.crossValidateModel(
                classifier,
                data,
                10,
                new Random(1)
            );
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }
    
}
