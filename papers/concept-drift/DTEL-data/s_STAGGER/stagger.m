function stagger(chunkSize, drift)
% color /in {r, b, g}
% shape /in {c, s, t}
% size  /in {s, m, l}
parameter.noiseRatio = 0.1;

data.chunkSize = chunkSize;
data.chunkNum = 120;
data.featureNum = 3;
data.classNum = 2;

trainXStr = randi([1 3], data.chunkSize * data.chunkNum, data.featureNum);
testXStr = randi([1 3], data.chunkSize * data.chunkNum, data.featureNum);
data.trainX = zeros(data.chunkSize * data.chunkNum, data.featureNum*2);
data.trainY = zeros(data.chunkSize * data.chunkNum, 1);
data.testX = zeros(data.chunkSize * data.chunkNum, data.featureNum*2);
data.testY = zeros(data.chunkSize * data.chunkNum, 1);
for i=1:data.chunkSize*data.chunkNum
    data.trainX(i, 1:2) = dec2bin(trainXStr(i, 1),2) - 48;
    data.trainX(i, 3:4) = dec2bin(trainXStr(i, 2),2) - 48;
    data.trainX(i, 5:6) = dec2bin(trainXStr(i, 3),2) - 48;
    data.testX(i, 1:2) = dec2bin(testXStr(i, 1),2) - 48;
    data.testX(i, 3:4) = dec2bin(testXStr(i, 2),2) - 48;
    data.testX(i, 5:6) = dec2bin(testXStr(i, 3),2) - 48;
end

for chunkCnt = 1:data.chunkNum
    conceptIndex = ceil(6*chunkCnt/data.chunkNum);
    dataStart = (chunkCnt - 1) * data.chunkSize + 1;
    dataEnd   = chunkCnt * data.chunkSize;
    for dataCnt = dataStart:dataEnd
        randTrain = rand(1,1);
        randTest = rand(1,1);
        if(drift==1)
            if(conceptIndex == 1)
                trainYTmp = (trainXStr(dataCnt, 1)==1)&&(trainXStr(dataCnt, 2)==1);
                testYTmp   = (testXStr(dataCnt, 1)==1) &&(testXStr(dataCnt, 2)==1);
            elseif(conceptIndex == 2)
                trainYTmp = (trainXStr(dataCnt, 1)==2)||(trainXStr(dataCnt, 2)==1);
                testYTmp   = (testXStr(dataCnt, 1)==2) ||(testXStr(dataCnt, 2)==1);
            elseif(conceptIndex == 3)
                trainYTmp = (trainXStr(dataCnt, 1)==3)||(trainXStr(dataCnt, 2)==2);
                testYTmp   = (testXStr(dataCnt, 1)==3) ||(testXStr(dataCnt, 2)==2);
            elseif(conceptIndex == 4)
                trainYTmp = (trainXStr(dataCnt, 1)==3)&&(trainXStr(dataCnt, 2)==3);
                testYTmp   = (testXStr(dataCnt, 1)==3) &&(testXStr(dataCnt, 2)==3);
            elseif(conceptIndex == 5)
                trainYTmp = (trainXStr(dataCnt, 1)==3)||(trainXStr(dataCnt, 2)==1);
                testYTmp   = (testXStr(dataCnt, 1)==3) ||(testXStr(dataCnt, 2)==1);
            elseif(conceptIndex == 6)
                trainYTmp = (trainXStr(dataCnt, 1)==1)||(trainXStr(dataCnt, 2)==2);
                testYTmp   = (testXStr(dataCnt, 1)==1) ||(testXStr(dataCnt, 2)==2);
            end
        elseif(drift==2)
            if(conceptIndex == 1)
                trainYTmp = (trainXStr(dataCnt, 1)==1)&&(trainXStr(dataCnt, 2)==1);
                testYTmp   = (testXStr(dataCnt, 1)==1) &&(testXStr(dataCnt, 2)==1);
            elseif(conceptIndex == 2)
                trainYTmp = (trainXStr(dataCnt, 1)==2)&&(trainXStr(dataCnt, 2)==1);
                testYTmp   = (testXStr(dataCnt, 1)==2) &&(testXStr(dataCnt, 2)==1);
            elseif(conceptIndex == 3)
                trainYTmp = (trainXStr(dataCnt, 1)==2)||(trainXStr(dataCnt, 2)==1);
                testYTmp   = (testXStr(dataCnt, 1)==2) ||(testXStr(dataCnt, 2)==1);
            elseif(conceptIndex == 4)
                trainYTmp = (trainXStr(dataCnt, 1)==2)||(trainXStr(dataCnt, 2)==2);
                testYTmp   = (testXStr(dataCnt, 1)==2) ||(testXStr(dataCnt, 2)==2);
            elseif(conceptIndex == 5)
                trainYTmp = (trainXStr(dataCnt, 1)==2)&&(trainXStr(dataCnt, 2)==2);
                testYTmp   = (testXStr(dataCnt, 1)==2) &&(testXStr(dataCnt, 2)==2);
            elseif(conceptIndex == 6)
                trainYTmp = (trainXStr(dataCnt, 1)==3)&&(trainXStr(dataCnt, 2)==2);
                testYTmp   = (testXStr(dataCnt, 1)==3) &&(testXStr(dataCnt, 2)==2);
            end
        end
        % train data
        if(randTrain > parameter.noiseRatio)
            data.trainY(dataCnt) = double(trainYTmp) + 1;
        else
            data.trainY(dataCnt) = double(~trainYTmp) + 1;
        end
        % test data
        if(randTest > parameter.noiseRatio)
            data.testY(dataCnt) = double(testYTmp) + 1;
        else
            data.testY(dataCnt) = double(~testYTmp) + 1;
        end
    end    
end

save('STA.mat', 'data');
end