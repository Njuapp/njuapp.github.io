function circle(chunkSize, drift)
parameter.noiseRatio = 0.1;
sigma = [3, 2, 1, 2, 3, 4, 5, 4;
         3, 2.5, 2, 2.5, 3, 3.5, 4, 3.5];

data.chunkSize = chunkSize;
data.chunkNum = 120;
data.featureNum = 3;
data.classNum = 2;

data.trainX = 10 * rand(data.chunkSize * data.chunkNum, data.featureNum) - 5;
data.testX  = 10 * rand(data.chunkSize * data.chunkNum, data.featureNum) - 5;
data.trainY = zeros(data.chunkSize * data.chunkNum, 1);
data.teseY = zeros(data.chunkSize * data.chunkNum, 1);

for chunkCnt = 1:data.chunkNum
    conceptIndex = ceil(8*chunkCnt/data.chunkNum);
    
    dataStart = (chunkCnt - 1) * data.chunkSize + 1;
    dataEnd   = chunkCnt * data.chunkSize;
    for dataCnt = dataStart:dataEnd
        randTrain = rand(1,1);
        randTest = rand(1,1);
        % train
        if(sum(data.trainX(dataCnt, 1:2).*data.trainX(dataCnt, 1:2)) < sigma(drift, conceptIndex)*sigma(drift, conceptIndex))
            if(randTrain > parameter.noiseRatio)
                data.trainY(dataCnt) = 1;
            else
                data.trainY(dataCnt) = 2;
            end
        else
            if(randTrain > parameter.noiseRatio)
                data.trainY(dataCnt) = 2;
            else
                data.trainY(dataCnt) = 1;
            end
        end
        % test
        if(sum(data.testX(dataCnt, 1:2).*data.testX(dataCnt, 1:2)) < sigma(drift, conceptIndex)*sigma(drift, conceptIndex))
            if(randTrain > parameter.noiseRatio)
                data.testY(dataCnt) = 1;
            else
                data.testY(dataCnt) = 2;
            end
        else
            if(randTrain > parameter.noiseRatio)
                data.testY(dataCnt) = 2;
            else
                data.testY(dataCnt) = 1;
            end
        end
    end
    
end
save('CIR.mat', 'data');
end

