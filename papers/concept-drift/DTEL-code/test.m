load('SEA200G.mat');
dataNum = length(testY);
chunkNum = dataNum / chunkSize;
accStat = zeros(chunkNum, 0);
for i = 1:chunkNum
    idx = ((i-1)* chunkSize + 1):i * chunkSize;
    accStat(i) = sum(classifyY(idx) == (testY(idx))')/chunkSize;
end
average = mean(accStat);
variance = std(accStat);