function [  ] = wekaOnlineClassification( X, Y )
%WEKAONLINECLASSIFICATION Summary of this function goes here
%   Detailed explanation goes here
    
    nFeatures = size(X, 2);
    nInstances = size(X, 1);
    
    featName = cell(1, nFeatures);
    for i = 1 : nFeatures
        featName{i} = num2str(i);
    end
%     import matlab2weka.*;

    %% Converting to WEKA data  
    disp('    Converting Data into WEKA format...');

    % Convert the training data to an Weka object
    % Java function: convert2weka(name, attrNameNumeric[], dataNumeric[][], classLabel[], hasClass)
    %
    %   hasClass: A boolean variable indicating whether or not to include the "class" 
    %       attribute (e.g., true for classification/regreesion, and false for clustering)
    
    convert2wekaObj = matlab2weka.ConvertToWeka('test', featName, X', Y, true); 
    dataWeka = convert2wekaObj.getInstances();
    clear convert2wekaObj;
    
    disp('    Converting Completed!');
    dataWeka.setClassIndex(dataWeka.numAttributes() - 1); 
    
    %% Online classification
    disp('    Online classification...');
    import weka.classifiers.trees.HoeffdingTree.*;
%     
%     % Create an java object
    model = weka.classifiers.trees.HoeffdingTree();
    model.buildClassifier(dataWeka);
    
    for i = 0 : nInstances - 1
        probs = model.distributionForInstance(dataWeka.get(i));
        model.updateClassifier(dataWeka.get(i));
    end
    
    disp('    Finish classification...');
end

