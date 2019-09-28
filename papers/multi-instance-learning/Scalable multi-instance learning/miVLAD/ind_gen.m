testIndex = zeros(100,254);
for i = 1:10
    testInd = randperm(2541);
    for j = 1:10
        testIndex((i-1)*10+j,:) = testInd( 1 + (j-1) * 254 : j * 254);
    end
end
save('../data/ngc/ngc1_Index.mat', 'testIndex');