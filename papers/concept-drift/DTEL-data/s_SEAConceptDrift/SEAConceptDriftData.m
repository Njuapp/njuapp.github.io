function SEAConceptDriftData()

parameter.noiseRatio = 0.1;

data.chunkSize = 200;
data.chunkNum = 120;
data.featureNum = 3;
data.classNum = 2;

data.trainX = zeros(data.chunkSize * data.chunkNum, data.featureNum);
data.trainY = zeros(data.chunkSize * data.chunkNum, 1);
data.testX = zeros(data.chunkSize * data.chunkNum, data.featureNum);
data.teseY = zeros(data.chunkSize * data.chunkNum, 1);

% generating: every 25 time steps change concept
% fea1 + fea2 < sigma  =>  class 1, else class 2 [ with noise, ratio = parameter.noiseRatio ]
% sigma = 8 -> 9 -> 7.5 -> 9.5
data.trainX = 10 * rand(data.chunkSize * data.chunkNum, data.featureNum);
data.testX  = 10 * rand(data.chunkSize * data.chunkNum, data.featureNum);
%sigma = [8, 9, 7.5, 9.5];
sigma = [10, 8, 6, 8, 10, 12, 14, 12];
for chunkCnt = 1:data.chunkNum
    conceptIndex = ceil(8*chunkCnt/data.chunkNum);
    
    dataStart = (chunkCnt - 1) * data.chunkSize + 1;
    dataEnd   = chunkCnt * data.chunkSize;
    
    for dataCnt = dataStart:dataEnd
        randTrain = rand(1,1);
        % train data
        if(sum(data.trainX(dataCnt, 1:2)) < sigma(conceptIndex))
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
        % test data
        if(sum(data.testX(dataCnt, 1:2)) < sigma(conceptIndex))
            data.testY(dataCnt) = 1;
        else
            data.testY(dataCnt) = 2;
        end
    end
end

save('SEA200G8.mat', 'data');

end