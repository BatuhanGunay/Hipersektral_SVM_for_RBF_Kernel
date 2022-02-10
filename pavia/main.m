% Prepare Dataset

clc; clear; close all;

datasetName='Pavia'; 
param.percent=95; 
param.features=false;
%dataset
[img,gt,X,numCls,trnIdx,tstIdx,numTrainEachCls,numTestEachCls]=funLoadClsData(datasetName, param);
[m,n,d]=size(img);
trnX=X.trnX;
trnY=X.trnY;
tstX=X.tstX;
tstY=X.tstY;
nTst=numel(tstIdx);

% model oluşturumu
gamma = 0.8;
t = templateSVM('KernelFunction','rbf');
mdlSVM = fitcecoc(trnX,trnY,'Learners',t);

%başarım hesaplamaları
Z_test = predict(mdlSVM,tstX);
cm=confusionmat(X.tstY,Z_test);
confusionchart(cm)
OA=(sum(diag(cm))/sum(cm(:)))*100; 
CA=(diag(cm)./sum(cm,2))*100; 
AA=mean(CA); %This is accuracy
cip=sum(cm,2);cpi=sum(cm)';
Pe= sum(cip.*cpi)/(nTst^2);
kappa= (.01*OA-Pe)/(1-Pe);
fprintf('OA=%1.2f AA=%1.2f kappa=%1.3f \n',OA,AA,kappa);
