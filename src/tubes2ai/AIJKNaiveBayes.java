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
import weka.core.CapabilitiesHandler;

/**
 *
 * @author Johan
 */
public class AIJKNaiveBayes implements Classifier, CapabilitiesHandler {

    private int[][][] freq;
    int nAttribute;
    int classIndex;
    private double[][][] prob;
    int nClassValue;

    
    @Override
    public void buildClassifier(Instances i) throws Exception {
//        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        nAttribute = i.numAttributes();
        classIndex = i.classIndex();
        Instance inst;
        Attribute att;
        int n_instance = i.numInstances();
        //inisialisasi matrix 3x3;
        //pertama cari ada berapa value di kelas
        nClassValue = i.attribute(classIndex).numValues();
        
        freq = new int[nAttribute][][];
        prob = new double[nAttribute][][];
        
        int a = 0;
        while(a < nAttribute){
            int nValue = i.attribute(a).numValues();
            if(a != classIndex){
                freq[a] = new int[nValue][nClassValue];
                prob[a] = new double[nValue][nClassValue];
            }else{
                freq[a] = new int[1][nClassValue];
                prob[a] = new double[1][nClassValue];
            }
            a++;
        }

        System.out.println("beres buat matriks");
        //inisialisasi matriks sama nilai 0
        a = 0; 
        int b;
        int c;
        while(a < nAttribute){ //outlook dkk
            b=0;
            int nValue = i.attribute(a).numValues();
            System.out.println("row "+a);
            while(b < nValue){
                c=0;
                System.out.println("row1 "+b);
                if(a==classIndex){
                        System.out.println("row2 "+c);
                        freq[a][0][b] = 0;
                }
                else {
                    while(c < nClassValue){
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
        int val;
        int classValue;
        while(a < n_instance){
            inst = i.get(a);
            b = 0;
            classValue = (int) inst.value(classIndex);
            while(b< nAttribute){
                val = (int) inst.value(b);
                if(b==classIndex){
                    freq[b][0][classValue]++;                
                }
                else {
                    freq[b][val][classValue]++;
                }
                b++;
            }
            a++;
        }       
        System.out.println("beres frekuensi!!!!");

        a=0;
        while(a < nAttribute){
            b = 0;
            int nValue = i.attribute(a).numValues();
            System.out.println("row "+a);
            while(b< nValue){
                System.out.println("row1 "+b);
                if(a!=classIndex){
                    c = 0;
                    while(c < nClassValue){
                        System.out.println("freq "+freq[a][b][c]);
                        System.out.println("freq_index "+freq[classIndex][0][c]);
                        prob[a][b][c] = (double) (freq[a][b][c]) / (double) (freq[classIndex][0][c]);
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
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        double[] probClass = new double[nClassValue];
        for(int x=0; x<nClassValue; x++){
            probClass[x] = 1;
        }
        
        int a=0;
        while(a < nAttribute){ //loop sebanyak atribut yg ada
            int b = 0;
            int val = (int) instnc.value(a);
            //System.out.println("row "+a);
            while(b< nClassValue){
                // System.out.println("row1 "+b);
                // System.out.println("String value = " + instnc.stringValue(b));
                if(a!=classIndex){ //atributnya diitung 
                    probClass[b] *= prob[a][val][b];
                }
                else {
                    probClass[b] *= prob[a][0][b];
                }
                b++;
            }
            a++;
        }
        
        a=0;
        int indexMax = 0;
        double max = probClass[indexMax];
        while(a<nClassValue){
            if(probClass[a] > max){
                indexMax = a;
                max = probClass[indexMax];
            }
            a++;
        }
        return indexMax;
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        double[] probClass = new double[nClassValue];
        for(int x=0; x<nClassValue; x++){
            probClass[x] = 1;
        }
        
        int a=0;
        int max=0;
        while(a < nAttribute){ //loop sebanyak atribut yg ada
            int b = 0;
            int val = (int) instnc.value(a);
            //System.out.println("row "+a);
                while(b< nClassValue){
                    // System.out.println("row1 "+b);
                    // System.out.println("String value = " + instnc.stringValue(b));
                    if(a!=classIndex){ //atributnya diitung 
                        probClass[b] *= prob[a][val][b];
                    }
                    else {
                        probClass[b] *= prob[a][0][b];
                    }
                    b++;
                }
            
            a++;
        }
        return probClass;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities c = new Capabilities((CapabilitiesHandler) this);
        c.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        c.enable(Capabilities.Capability.NOMINAL_CLASS);
        return c;        
    }
    
}
