function [reserveModel, predictModel] = modelTrainDT(x, y)
    predictModel = fitctree(x, y, 'MergeLeaves', 'off');
    reserveModel = prune(predictModel);
end