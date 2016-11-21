/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

/**
 *
 * @author khrs
 */
public class DriverNB {
    public static void run(String data) throws Exception{
        System.out.println("tes driver");

        ConverterUtils.DataSource source = new ConverterUtils.DataSource(data);
        Instances dataTrain = source.getDataSet();
        //if (dataTrain.classIndex() == -1)
           dataTrain.setClassIndex(0);
        ArffSaver saver = new ArffSaver();

//        dataTrain.setClassIndex();
        Discretize discretize = new Discretize();
        discretize.setInputFormat(dataTrain);
        Instances dataTrainDisc = Filter.useFilter(dataTrain, discretize);
        
        //NaiveBayes NB = new NaiveBayes();
        AIJKNaiveBayes NB = new AIJKNaiveBayes();
        NB.buildClassifier(dataTrainDisc);
        
        Evaluation eval = new Evaluation(dataTrainDisc);
        eval.evaluateModel(NB, dataTrainDisc);
        
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
        /*Instance inst = new DenseInstance(5);
        
        inst.setDataset(dataTrain);
        inst.setValue(0, "sunny");
        inst.setValue(1, "hot");
        inst.setValue(2, "high");
        inst.setValue(3, "FALSE");
        inst.setValue(4, "yes");
        double a = NB.classifyInstance(inst);
        String hasil="";
        if(a==0.0){
            hasil="YES";
        } else{
            hasil="NO";
        }
//double[] b = NB.distributionForInstance(inst);
        System.out.println("Hasil klasifikasi: "+hasil);
        //System.out.println(b);*/
    }
}
