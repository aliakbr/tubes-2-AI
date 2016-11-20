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

    private int[][][] freq;
    int n_attrib;
    int index_class;
    private double[][][] prob;
    

    
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
        
        freq = new int[n_attrib][][];
        prob = new double[n_attrib][][];
        
        int a = 0;
        while(a < n_attrib){
            int n_val = i.attribute(a).numValues();
            if(a != index_class){
                freq[a] = new int[n_val][n_value_class];
                prob[a] = new double[n_val][n_value_class];
            }else{
                freq[a] = new int[1][n_value_class];
                prob[a] = new double[1][n_value_class];
            }
            a++;
        }

        System.out.println("beres buat matriks");
        //inisialisasi matriks sama nilai 0
        a = 0; 
        int b=0;
        int c=0;
        while(a < n_attrib){ //outlook dkk
            b=0;
            int n_val = i.attribute(a).numValues();
            System.out.println("row "+a);
            while(b < n_val){
                c=0;
                System.out.println("row1 "+b);
                if(a==index_class){
                        System.out.println("row2 "+c);
                        freq[a][0][b] = 0;
                }
                else {
                    while(c < n_value_class){
                        System.out.println("row2 "+c);
                        freq[a][b][c] = 0;
                        c++;
                    }
                }
                b++;
            }
            a++;
        }


        System.out.println("beres inisialisasi 0");
        
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
                    freq[b][0][class_val]++;                
                }
                else {
                    freq[b][val][class_val]++;
                }
                b++;
            }
            a++;
        }       
        System.out.println("beres frekuensi!!!!");

        a=0;
        while(a < n_attrib){
            b = 0;
            int n_val = i.attribute(a).numValues();
            System.out.println("row "+a);
            while(b< n_val){
                System.out.println("row1 "+b);
                if(a!=index_class){
                    c = 0;
                    while(c < n_value_class){
                        System.out.println("freq "+freq[a][b][c]);
                        System.out.println("freq_index "+freq[index_class][0][c]);
                        prob[a][b][c] = (double) (freq[a][b][c]) / (double) (freq[index_class][0][c]);
                        System.out.println("prob ["+a+"]["+b+"]["+c+"] "+ prob[a][b][c]);
                        c++;
                    }
                }
                else {
                    prob[a][0][b] = (double) freq[a][0][b] / i.numInstances();
                    System.out.println("prob ["+a+"][0]["+b+"] "+ prob[a][0][b]);
                }
                b++;
            }
            a++;
        }        
        System.out.println("beres prob!!!!");

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
