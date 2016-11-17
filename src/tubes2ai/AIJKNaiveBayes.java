/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes2ai;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author i
 */
public class AIJKNaiveBayes implements Classifier {

    private int[][][] attrib_sum;
    int n_attrib;
    int index_class;
//    Map<String, Double>[] valToIndex;
    /* ngelist ada berapa atribut
    n_attrib = 
    n_class;
    int[][][] = new int[x][][];
    
    for each attrrib{
        ada berapa nilai
        buat fungsi hash
        attrib[i] = new int[j][n_class];
    }
    
    
    for each instance {
        kelas = scan kelas;
        for each attrib {
            scan value;
            int[attrib][value][kelas]++; 
        }
    }
    
    
    private in
    */
    
    @Override
    public void buildClassifier(Instances i) throws Exception {
//        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        n_attrib = i.numAttributes();
        index_class = i.classIndex();
        Instance inst;
        Attribute att;
        int n_instance = i.numInstances();
        //inisialisasi matrix 3x3;
        //pertama cari ada berapa value di kelas
        int n_value_class = i.attribute(index_class).numValues();
        
        attrib_sum = new int[n_attrib][][];
        
        int a = 0;
        while(a < n_attrib){
            int n_val = i.attribute(a).numValues();
            if(a != index_class)
                attrib_sum[a] = new int[n_val][n_value_class];
            else
                attrib_sum[a] = new int[1][n_value_class];
        }

        //inisialisasi matriks sama nilai 0
        a = 0;
        int b=0;
        int c=0;
        while(a < n_attrib){
            b=0;
            int n_val = i.attribute(a).numValues();
            while(b < n_val){
                c=0;
                while(c < index_class){
                    attrib_sum[a][b][c] = 0;
                }
            }
        }
        
        a = 0;
        b = 0;
        int val;
        int class_val;
        while(a < n_instance){
            inst = i.get(a);
            b = 0;
            class_val = (int) inst.value(index_class);
            while(b< n_attrib){
                val = (int) inst.value(b);
                if(b==index_class){
                    attrib_sum[b][0][class_val]++;                
                }
                else {
                    attrib_sum[b][val][class_val]++;
                }
            }
        }       
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
