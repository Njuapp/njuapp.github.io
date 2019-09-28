function svm2mat()
feaMax = 0;
dataSize = 100000;
feaMaxK = 780434;

%xStream = zeros(30*dataSize, feaMaxK);
%yStream = zeros(30*dataSize, 1);

for i=1:30
    filePath = ['SameDimData/N201509', sprintf('%02d',i)];
    [y, x] = libsvmread(filePath);
    
    [~, feaTmp] = size(x);
    if(feaTmp > feaMax)
        feaMax = feaTmp;
    end
    
%     x = zeros(dataSize, feaMaxK);
%     x(:, 1:feaTmp) = xTmp;
%     y = yTmp;
    save(['201509', sprintf('%02d',i), '.mat'], 'x', 'y');
    
    %startI = dataSize*(i-1) + 1;
    %endI   = dataSize*i;
    %xStream(startI:endI, :) = x;
    %yStream(startI:endI)    = y;
end
disp(feaMax);
    %save('TecentStream.mat', 'xStream', 'yStream');
end