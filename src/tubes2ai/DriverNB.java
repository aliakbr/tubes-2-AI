/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author khrs
 */
public class DriverNB {
    public static void run() throws Exception{
        System.out.println("tes driver");


        ConverterUtils.DataSource source = new ConverterUtils.DataSource("weather.nominal.arff");
        Instances dataTrain = source.getDataSet();
        if (dataTrain.classIndex() == -1)
           dataTrain.setClassIndex(dataTrain.numAttributes() -1);
        ArffSaver saver = new ArffSaver();

        dataTrain.setClassIndex(dataTrain.numAttributes()-1);

        AIJKNaiveBayes NB = new AIJKNaiveBayes();
        NB.buildClassifier(dataTrain);
        Instance inst = new DenseInstance(5);
        
        inst.setDataset(dataTrain);
        inst.setValue(0, "overcast");
        inst.setValue(1, "hot");
        inst.setValue(2, "high");
        inst.setValue(3, "FALSE");
        inst.setValue(4, "no");
        double a = NB.classifyInstance(inst);
        System.out.println("Hasil klasifikasi: "+a);
    }
}
