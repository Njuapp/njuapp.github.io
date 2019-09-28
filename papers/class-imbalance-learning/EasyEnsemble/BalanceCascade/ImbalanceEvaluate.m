function [f,g]=ImbalanceEvaluate(target,result,pclass,nclass)

% F-measure and G-mean

tp=sum(target==pclass & result==pclass);
fn=sum(target==pclass & result==nclass);
tn=sum(target==nclass & result==nclass);
fp=sum(target==nclass & result==pclass);

if(tp==0)    
    f=0;
    g=0;
else
    precision=tp/(tp+fp);
    recall=tp/(tp+fn);
    f=2*(precision*recall)/(precision+recall);

    pacc=recall;
    nacc=tn/(tn+fp);
    g=sqrt(pacc*nacc);
end





