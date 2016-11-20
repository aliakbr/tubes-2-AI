package tubes2ai;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Standardize;

import java.util.Random;

/**
 * Created by i on 2016-11-20.
 */
public class DriverFFNN {
    public static void main(String[] args) {
        ConverterUtils.DataSource source;

        try {
            AIJKFFNN classifier = new AIJKFFNN();
            classifier.setOptions(Utils.splitOptions("-L 1 -N 30"));
            source = new ConverterUtils.DataSource("data/iris.arff");
            Instances rawData = source.getDataSet();
            rawData.setClassIndex(rawData.numAttributes() - 1);

            Filter filter = new MultiFilter();
            filter.setOptions(Utils.splitOptions("-F \"weka.filters.unsupervised.attribute.Normalize -S 1.0 -T 0.0\" -F \"weka.filters.unsupervised.attribute.Standardize \""));
            filter.setInputFormat(rawData);
            Instances data = Filter.useFilter(rawData, filter);

            Evaluation evaluation = new Evaluation(data);
            evaluation.crossValidateModel(classifier, data, 10, new Random(1));
            System.out.println(evaluation.toSummaryString());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
