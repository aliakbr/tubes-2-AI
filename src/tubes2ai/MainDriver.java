package tubes2ai;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.sun.org.apache.xpath.internal.SourceTree;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.Discretize;

import java.io.Serializable;
import java.util.Random;

/**
 * Created by i on 2016-11-21.
 */
public class MainDriver {
    @Parameter(names = {"--ffnn"})
    private boolean useFFNN = false;

    @Parameter(names = {"--nb"})
    private boolean useNB = false;

    @Parameter(names = {"--model"})
    private String modelFilename = "";

    @Parameter(names = {"--cross-validate"})
    private Integer cvFold = 0;
    
    @Parameter(names = {"--split-test"})
    private Integer splitTest = 0;

    @Parameter(names = {"-f"}, required = true)
    private String filename;

    @Parameter(names = {"--class-index"})
    private Integer classIndex = null;

    @Parameter(names = {"--remove"})
    private Integer classIndexRemoved = null;

    @Parameter(names = {"--options"})
    private String classifierOptions = "";

    public static void main(String[] args) {
        MainDriver mainDriver = new MainDriver();
        new JCommander(mainDriver, args);

        mainDriver.run();
    }

    private void run() {
        try {
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(filename);
            Instances dataTrain = source.getDataSet();
            dataTrain.randomize(new Random(1337));
            Classifier classifier;
            Instances filteredData;
            System.out.println(modelFilename);
            Filter usedFilter = null;
            dataTrain.setClassIndex((classIndex != null) ? classIndex : (dataTrain.numAttributes() - 1));
            if (classIndexRemoved != null){
                dataTrain.deleteAttributeAt(classIndexRemoved);
            }
            /*
            if (classIndexRemoved != null){
                Filter filterX = new MultiFilter();
                filterX.setOptions(Utils.splitOptions(
                    "-F \"weka.filters.unsupervised.attribute.Remove -R " + classIndexRemoved + "\""
                ));
                filterX.setInputFormat(dataTrain);
                filteredData1 = Filter.useFilter(dataTrain,filterX);
            }
            for (int i = 0; i < filteredData1.numAttributes(); i++) {
                System.out.println(filteredData1.attribute(i).toString());
            }*/

            if (useNB) {
                AIJKNaiveBayes NB = new AIJKNaiveBayes();
                classifier = NB;
                /* Discretize discretize = new Discretize();
                discretize.setInputFormat(dataTrain);
                usedFilter = discretize;
                filteredData = Filter.useFilter(dataTrain, discretize);*/
                Filter filter1 = new MultiFilter();
                if (classIndexRemoved == null) {
                    filter1.setOptions(Utils.splitOptions(
                            "-F \"weka.filters.unsupervised.attribute.Discretize\""
                    ));
                }
                else{
                    filter1.setOptions(Utils.splitOptions(
                            "-F \"weka.filters.unsupervised.attribute.Discretize\""
                         //   "-F \"weka.filters.unsupervised.attribute.Remove -R " + classIndexRemoved + "\""
                    ));
                }
                filter1.setInputFormat(dataTrain);
                usedFilter = filter1;
                filteredData = Filter.useFilter(dataTrain, filter1);
            } else if (useFFNN) {
                AIJKFFNN FFNN = new AIJKFFNN();
                FFNN.setOptions(Utils.splitOptions(classifierOptions));
                classifier = FFNN;

                Filter filter = new MultiFilter();
                if (classIndexRemoved == null) {
                    filter.setOptions(Utils.splitOptions(
                            "-F \"weka.filters.unsupervised.attribute.ReplaceMissingValues\"" +
                            "-F \"weka.filters.supervised.attribute.NominalToBinary\"" +
                            "-F \"weka.filters.unsupervised.attribute.Normalize -S 1.0 -T 0.0\"" +
                            "-F \"weka.filters.unsupervised.attribute.Standardize \""
                    ));
                }
                else{
                    filter.setOptions(Utils.splitOptions(
                            "-F \"weka.filters.unsupervised.attribute.ReplaceMissingValues\"" +
                            "-F \"weka.filters.supervised.attribute.NominalToBinary\"" +
                            "-F \"weka.filters.unsupervised.attribute.Normalize -S 1.0 -T 0.0\"" +
                            "-F \"weka.filters.unsupervised.attribute.Standardize \""
                    ));
                }
                filter.setInputFormat(dataTrain);
                usedFilter = filter;
                filteredData = Filter.useFilter(dataTrain, filter);
            } else if (modelFilename.length() > 0) {
                CF cf = (CF) SerializationHelper.read(modelFilename);
                filteredData = Filter.useFilter(dataTrain, cf.f);
                classifier = cf.c;
            } else {
                throw new RuntimeException("Need to pick a classification method");
            }
            Evaluation evaluation = new Evaluation(filteredData);

            long evalStartTime = System.currentTimeMillis();

            if (useFFNN || useNB) {
                if (cvFold <= 0) {
                    classifier.buildClassifier(filteredData);
                    if (modelFilename.length() > 0) {
                        CF cf = new CF();
                        cf.c = classifier;
                        cf.f = usedFilter;
                        SerializationHelper.write(modelFilename, cf);
                    }
                    evaluation.evaluateModel(classifier, filteredData);
                } else {
                    evaluation.crossValidateModel(classifier, filteredData, cvFold, new Random(1));
                    classifier.buildClassifier(filteredData);
                    if (modelFilename.length() > 0) {
                        CF cf = new CF();
                        cf.c = classifier;
                        cf.f = usedFilter;
                        SerializationHelper.write(modelFilename, cf);
                    }
                }
                if (cvFold > 0) {
                    evaluation.crossValidateModel(classifier, filteredData, cvFold, new Random(1));
                    classifier.buildClassifier(filteredData);
                    if (modelFilename.length() > 0) {
                        CF cf = new CF();
                        cf.c = classifier;
                        cf.f = usedFilter;
                        SerializationHelper.write(modelFilename, cf);
                    }
                } else if (splitTest > 0){
                    int trainSize = (int) Math.round(filteredData.numInstances() * splitTest/100);
                    int testSize = filteredData.numInstances() - trainSize;
                    Instances train = new Instances(filteredData, 0, trainSize);
                    Instances test = new Instances(filteredData, trainSize, testSize);
                    classifier.buildClassifier(train);
                    if (modelFilename.length() > 0) {
                        CF cf = new CF();
                        cf.c = classifier;
                        cf.f = usedFilter;
                        SerializationHelper.write(modelFilename, cf);
                    }
                    evaluation.evaluateModel(classifier, test);
                } else {
                    classifier.buildClassifier(filteredData);
                    if (modelFilename.length() > 0) {
                        CF cf = new CF();
                        cf.c = classifier;
                        cf.f = usedFilter;
                        SerializationHelper.write(modelFilename, cf);
                    }
                    evaluation.evaluateModel(classifier, filteredData);
                }
            } else {
                evaluation.evaluateModel(classifier, filteredData);
            }

            long evalTime = System.currentTimeMillis() - evalStartTime;

            System.out.println(evaluation.toSummaryString());
            System.out.println(evaluation.toClassDetailsString());
            System.out.println(evaluation.toMatrixString());
            System.out.printf("Total time: %d ms%n", evalTime);

        } catch (Exception e) {
            throw new RuntimeException(e);
        }



    }
}
