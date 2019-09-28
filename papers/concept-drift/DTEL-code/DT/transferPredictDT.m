function [yhat, yProb] = transferPredictDT(model, x)
    nodeIndex = 1;
    while(model.IsBranchNode(nodeIndex)==1)
        feaTmp = model.CutPredictor(nodeIndex);
        valTmp = model.CutPoint(nodeIndex);
        if(x(feaTmp) < valTmp)
            nodeIndex = model.Children(nodeIndex, 1);
        else
            nodeIndex = model.Children(nodeIndex, 2);
        end
    end
    yhat = model.NodeClass(nodeIndex);
    yProb = model.ClassProbability(nodeIndex);
end

