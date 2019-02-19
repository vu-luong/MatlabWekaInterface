clear all; close all;

%% Initializing 
% adding the path to matlab2weka codes
addpath('matlab2weka');

javaaddpath('matlab2weka/matlab2weka.jar')
javaaddpath('weka.jar')

%% Loading Iris Dataset
load iris

% numerical class variable
X = iris(:,1:4);
Y = iris(:,5);

% converting to nominal variables (Weka cannot classify numerical classes)
YNom = cell(size(Y));
uClassNum = unique(Y);
tmpCell = cell(1,1);
for i = 1:length(uClassNum)
    tmpCell{1,1} = strcat('class_', num2str(i-1));
    YNom(Y == uClassNum(i),:) = repmat(tmpCell, sum(Y == uClassNum(i)), 1);
end
clear uClass_num tmp_cell i

wekaOnlineClassification(X, YNom);

