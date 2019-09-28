function sine(chunkSize, drift)
parameter.noiseRatio = 0.1;

data.chunkSize = chunkSize;
data.chunkNum = 120;
data.featureNum = 2;
data.classNum = 2;

data.trainX = 10 * rand(data.chunkSize * data.chunkNum, data.featureNum) - 5;
data.trainY = zeros(data.chunkSize * data.chunkNum, 1);
data.testX = 10 * rand(data.chunkSize * data.chunkNum, data.featureNum) - 5;
data.testY = zeros(data.chunkSize * data.chunkNum, 1);

%%%
% 5*sin(x1+theta) </> x2
%%%
theta = [pi/30, pi/60];
for chunkCnt = 1:data.chunkNum
    conceptIndex = chunkCnt - 1;
    conceptTmp = theta(drift)*conceptIndex;
    dataStart = (chunkCnt - 1) * data.chunkSize + 1;
    dataEnd   = chunkCnt * data.chunkSize;
    for dataCnt = dataStart:dataEnd
        randTrain = rand(1,1);
        randTest = rand(1,1);
        % train data
        if(5*sin(data.trainX(dataCnt, 1)+conceptTmp) < data.trainX(dataCnt, 2))
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
        if(5*sin(data.testX(dataCnt, 1)+conceptTmp) < data.testX(dataCnt, 2))
            if(randTest > parameter.noiseRatio)
                data.testY(dataCnt) = 1;
            else
                data.testY(dataCnt) = 2;
            end
        else
            if(randTest > parameter.noiseRatio)
                data.testY(dataCnt) = 2;
            else
                data.testY(dataCnt) = 1;
            end
        end
    end
end
save('SIN.mat', 'data');
end

