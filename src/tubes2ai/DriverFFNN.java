package tubes2ai;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Standardize;

import java.io.IOException;
import java.util.*;

/**
 * Created by i on 2016-11-20.
 */
public class DriverFFNN {
    public static void main(String[] args) {
        ConverterUtils.DataSource source;

        List<String> argsList = Arrays.asList(args);

        boolean waitForInput = argsList.indexOf("-w") != -1;
        boolean crossValidate = argsList.indexOf("-c") != -1;
        String dataFile = argsList.get(argsList.size() - 1);

        if (waitForInput) {
            System.out.println("Press enter to start...");
            try {
                System.in.read();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            System.out.println("Starting...");
        }

        try {
            AIJKFFNN classifier = new AIJKFFNN();
            classifier.setOptions(Utils.splitOptions("-I 1000 -L 1 -N 20"));
            source = new ConverterUtils.DataSource(dataFile);
            Instances rawData = source.getDataSet();
            rawData.randomize(new Random(17));
            rawData.setClassIndex(rawData.numAttributes() - 1);

            Filter filter = new MultiFilter();
            filter.setOptions(Utils.splitOptions(
                    "-F \"weka.filters.unsupervised.attribute.ReplaceMissingValues\"" +
                    "-F \"weka.filters.supervised.attribute.NominalToBinary\"" +
                    "-F \"weka.filters.unsupervised.attribute.Normalize -S 1.0 -T 0.0\"" +
                    "-F \"weka.filters.unsupervised.attribute.Standardize \""
            ));
            filter.setInputFormat(rawData);
            Instances data = Filter.useFilter(rawData, filter);

            Evaluation evaluation = new Evaluation(data);

            if (crossValidate) {
                evaluation.crossValidateModel(classifier, data, 10, new Random(1));
            } else {
                classifier.buildClassifier(data);
                evaluation.evaluateModel(classifier, data);
            }

            System.out.println(evaluation.toSummaryString());
            System.out.println(evaluation.toMatrixString());


        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        if (waitForInput) {
            System.out.println("Finished, press enter to exit...");
            try {
                System.in.read();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            System.out.println("Exiting...");
        }
    }
}
