package matlab2weka;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Test {

    public static void main(String[] args) throws IOException {

        BufferedReader reader =
                new BufferedReader(new FileReader("/Users/AnhVu/Study/PhD/weka_matlab(19feb2019)/MatlabWekaInterface/java/iris.arff"));
        ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
        Instances data = arff.getData();
        data.setClassIndex(data.numAttributes() - 1);

        System.out.println(data.stringFreeStructure());

    }
}
