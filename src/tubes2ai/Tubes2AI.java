/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author i
 */
public class Tubes2AI {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        System.out.println("tes");
        
        
        DataSource source = new DataSource("iris.arff");
        Instances dataTrain = source.getDataSet();
        if (dataTrain.classIndex() == -1)
           dataTrain.setClassIndex(dataTrain.numAttributes() -1);
        ArffSaver saver = new ArffSaver();

        AIJKNaiveBayes NB = new AIJKNaiveBayes();
//        Instances
//        NB.buildClassifier();
    }
    
}
