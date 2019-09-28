function sampleIdx = EasyEnsemble(possize, negsize, samples)
    if possize > negsize
        negIdx = repmat(1:negsize, 1, samples);
        negIdx = negIdx + possize;
        posIdx = zeros(1, samples * negsize);
        for ii = 1: samples
            posIdx((ii-1)* negsize + 1 : ii* negsize) = randperm(possize, negsize);
        end
        sampleIdx = [posIdx, negIdx];
    else
        posIdx = repmat(1:possize, 1, samples);
        negIdx = zeros(1, samples * possize);
        for ii = 1: samples
            negIdx((ii-1)* possize + 1: ii* possize) = randperm(negsize, possize);
        end
        negIdx = negIdx + possize;
        sampleIdx = [posIdx, negIdx];
    end
end