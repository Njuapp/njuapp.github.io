function RotateConceptDriftData()

load('fourclass/fourclass48000.mat');

parameter.noiseRatio = 0.1;
parameter.circleCnt  = 1;

data.chunkSize = 200;
data.chunkNum = 120;
data.featureNum = 2;
data.classNum = 2;

data.trainX = x(1:(data.chunkSize * data.chunkNum), :);
data.trainY = y(1:(data.chunkSize * data.chunkNum), :);
data.testX = x((data.chunkSize * data.chunkNum + 1):end, :);
data.testY = y((data.chunkSize * data.chunkNum + 1):end, :);

%%%
% x' = x * cos(theta) - y * sin(theta)
% y' = x * sin(theta) + y * cos(theta)
%%%
for chunkCnt = 1:data.chunkNum
    theta = parameter.circleCnt * 4 * pi * chunkCnt / data.chunkSize;
    sinThe = sin(theta);
    cosThe = cos(theta);
    
    dataStart = (chunkCnt - 1) * data.chunkSize + 1;
    dataEnd   = chunkCnt * data.chunkSize;
    for dataCnt = dataStart:dataEnd
        data.trainX(dataCnt, 1) = data.trainX(dataCnt, 1) * cosThe - data.trainX(dataCnt, 2) * sinThe;
        data.trainX(dataCnt, 2) = data.trainX(dataCnt, 1) * sinThe + data.trainX(dataCnt, 2) * cosThe;
        randTrain = rand(1,1);
        if(randTrain < parameter.noiseRatio)
            noiseY = randint(1,1,[1 (data.classNum-1)]);
            if(noiseY >= data.trainY(dataCnt))
                noiseY = noiseY + 1;
            end
            
            data.trainY(dataCnt) = noiseY;
        end
        
        data.testX(dataCnt, 1) = data.testX(dataCnt, 1) * cosThe - data.testX(dataCnt, 2) * sinThe;
        data.testX(dataCnt, 2) = data.testX(dataCnt, 1) * sinThe + data.testX(dataCnt, 2) * cosThe;
    end
end

save('RotateConceptData.mat', 'data');

end

