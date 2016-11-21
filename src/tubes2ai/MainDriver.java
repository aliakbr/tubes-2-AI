package tubes2ai;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.Discretize;

import java.util.Random;

/**
 * Created by i on 2016-11-21.
 */
public class MainDriver {
    @Parameter(names = {"--ffnn"})
    private boolean useFFNN = false;

    @Parameter(names = {"--nb"})
    private boolean useNB = false;

    @Parameter(names = {"--cross-validate"})
    private Integer cvFold = 0;

    @Parameter(names = {"-f"}, required = true)
    private String filename;

    @Parameter(names = {"--class-index"})
    private Integer classIndex = null;

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

            dataTrain.setClassIndex((classIndex != null) ? classIndex : (dataTrain.numAttributes() - 1));

            if (useNB) {
                AIJKNaiveBayes NB = new AIJKNaiveBayes();
                classifier = NB;

                Discretize discretize = new Discretize();
                discretize.setInputFormat(dataTrain);
                filteredData = Filter.useFilter(dataTrain, discretize);
            } else if (useFFNN) {
                AIJKFFNN FFNN = new AIJKFFNN();
                FFNN.setOptions(Utils.splitOptions(classifierOptions));
                classifier = FFNN;

                Filter filter = new MultiFilter();
                filter.setOptions(Utils.splitOptions(
                        "-F \"weka.filters.unsupervised.attribute.ReplaceMissingValues\"" +
                        "-F \"weka.filters.supervised.attribute.NominalToBinary\"" +
                        "-F \"weka.filters.unsupervised.attribute.Normalize -S 1.0 -T 0.0\"" +
                        "-F \"weka.filters.unsupervised.attribute.Standardize \""
                ));
                filter.setInputFormat(dataTrain);
                filteredData = Filter.useFilter(dataTrain, filter);
            } else {
                throw new RuntimeException("Need to pick a classification method");
            }

            Evaluation evaluation = new Evaluation(filteredData);

            if (cvFold <= 0) {
                classifier.buildClassifier(filteredData);
                evaluation.evaluateModel(classifier, filteredData);
            } else {
                evaluation.crossValidateModel(classifier, filteredData, cvFold, new Random(1));
            }

            System.out.println(evaluation.toSummaryString());
            System.out.println(evaluation.toClassDetailsString());
            System.out.println(evaluation.toMatrixString());

        } catch (Exception e) {
            throw new RuntimeException(e);
        }



    }
}
