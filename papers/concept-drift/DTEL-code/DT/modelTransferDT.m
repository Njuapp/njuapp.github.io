function modelTransfer = modelTransferDT(model, x, y, tlflag)
    modelTransfer.Children = model.Children;                    % children node index
    modelTransfer.CutPredictor = char(model.CutPredictor);      % cut feature
    modelTransfer.CutPredictor(:, 1) = '0';
    modelTransfer.CutPredictor = str2num(modelTransfer.CutPredictor);
    modelTransfer.CutPoint = model.CutPoint;                    % cut point
    modelTransfer.NumNodes = model.NumNodes;                    % number of nodes
    modelTransfer.NodeClass = char(model.NodeClass);            % class of node
    modelTransfer.NodeClass = str2num(modelTransfer.NodeClass);
    modelTransfer.IsBranchNode = model.IsBranchNode;            % branch node of leaf
    modelTransfer.ClassProbability = max(model.ClassProbability, [], 2);
    if(tlflag == 0)
        return;
    end

    % data into nodes
    [dataTransferNum, ~] = size(x);
    nodeArray = zeros(dataTransferNum, 1);
    for dataCntTmp = 1:dataTransferNum
        nodeIndex = 1;
        while(modelTransfer.IsBranchNode(nodeIndex) == 1)
            cutFeatureIndex = modelTransfer.CutPredictor(nodeIndex);
            cutFeatureValue = modelTransfer.CutPoint(nodeIndex);
            if(x(dataCntTmp, cutFeatureIndex) < cutFeatureValue)
                nodeIndex = modelTransfer.Children(nodeIndex, 1);
            else
                nodeIndex = modelTransfer.Children(nodeIndex, 2);
            end
        end
        nodeArray(dataCntTmp) = nodeIndex;
    end
    % update nodes
    originalNumNodes = modelTransfer.NumNodes;
    for nodeCntTmp = 1:originalNumNodes
        if(modelTransfer.IsBranchNode(nodeCntTmp) == 0)
            dataNodeTmp = find(nodeArray == nodeCntTmp);
            if(~isempty(dataNodeTmp))
                inNodeNum = length(dataNodeTmp);
                inNodeX = x(dataNodeTmp,:);
                inNodeY = y(dataNodeTmp);
                if(length(unique(inNodeY))==1)
                    modelTransfer.NodeClass(nodeCntTmp) = inNodeY(1);
                elseif(inNodeNum < 10)
                    modelTransfer.NodeClass(nodeCntTmp) = mode(inNodeY);
                else
                    treeTmp = fitctree(inNodeX, inNodeY);
                    if(treeTmp.NumNodes == 1)
                        modelTransfer.NodeClass(nodeCntTmp) = str2double(char(treeTmp.NodeClass(1)));
                    else
                        CutPredictorTmp = char(treeTmp.CutPredictor);      % cut feature
                        CutPredictorTmp(:, 1) = '0';
                        CutPredictorTmp = str2num(CutPredictorTmp);
                        NodeClassTmp = char(treeTmp.NodeClass);            % class of node
                        NodeClassTmp = str2num(NodeClassTmp);

                        modelTransfer.Children(nodeCntTmp, :) = treeTmp.Children(1, :) + modelTransfer.NumNodes - 1;
                        modelTransfer.CutPredictor(nodeCntTmp) = CutPredictorTmp(1);
                        modelTransfer.CutPoint(nodeCntTmp) = treeTmp.CutPoint(1);
                        modelTransfer.NodeClass(nodeCntTmp) = NodeClassTmp(1);
                        modelTransfer.IsBranchNode(nodeCntTmp) = treeTmp.IsBranchNode(1);

                        modelTransfer.Children = [modelTransfer.Children; treeTmp.Children(2:end,:)+modelTransfer.NumNodes-1];
                        modelTransfer.CutPredictor = [modelTransfer.CutPredictor; CutPredictorTmp(2:end)];
                        modelTransfer.CutPoint = [modelTransfer.CutPoint; treeTmp.CutPoint(2:end)];
                        modelTransfer.NodeClass = [modelTransfer.NodeClass; NodeClassTmp(2:end)];
                        modelTransfer.IsBranchNode = [modelTransfer.IsBranchNode; treeTmp.IsBranchNode(2:end)];
                        modelTransfer.ClassProbability = [modelTransfer.ClassProbability; max(treeTmp.ClassProbability, [], 2)];

                        modelTransfer.NumNodes = modelTransfer.NumNodes + treeTmp.NumNodes - 1;
                    end
                end
            end
        end
    end
end