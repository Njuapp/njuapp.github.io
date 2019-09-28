clc
clear

%-------------------算法超参数--------------
chunkSize=50;
transfer_rate=5;
%--------------------读取数据集-------------
folderName='data_sea';
dataName={'SEA200A','SEA200G','SEA500G','CIR200A','CIR200G','CIR500G','SIN200A',...
   'SIN200G','SIN500G','STA200A','STA200G','STA500G'};
%dataName={'CIR200G'};
dataNum=size(dataName,2);
acc_record=zeros(dataNum,5);
Plot_recorder=zeros(dataNum,1200);
counter=1;
for i=7:dataNum
    load(['./',folderName,'/',dataName{i},'.mat']);   
%---------------------对训练集做划分--------
    x_train=data.trainX;
    y_train=data.trainY;
    x_test=data.testX;
    y_test=data.testY;
    
    data_num=size(x_train,1);
    test_num=floor(data_num*0.8);
    random_range=data_num-test_num;
    slice_size=floor(random_range/5);
    
%     for j=0:4

%         start_point=j*slice_size+1;
%         end_point=start_point+test_num-1;
%         if(end_point>data_num)
%              out='ERROR'
%         end

        X=x_train;
        y=y_train;    

%------------训练T轮-----------------
        T=floor(size(X,1)/chunkSize);
        P=floor(T);
        if P>25
            P=25;
        end
%         P=floor(T);
%         if P>2
%             P=2;
%         end
        buffer=cell(1,P+1);
        buffer_index=0;
        full=0;
        acc=0;
        for t=1:T-1
            if rem(t,floor(T/20))==0
                fprintf('%d  ',t)
            end
            
            start_pointT=t*chunkSize+1;
            end_pointT=start_pointT+chunkSize-1;
            if(end_pointT>size(X,1))
               out='ERROR'
            end
            X_t=X(start_pointT:end_pointT,:);
            y_t=y(start_pointT:end_pointT,:);
            Xt_test=x_test(start_pointT:end_pointT,:);
            yt_test=y_test(start_pointT:end_pointT,:);
            
%----------利用上一轮模型来预测本轮正确率----------------
            
%             if t==1
%                 prediction_now=predict(model_previous,X_t);
%                 acc_temp=sum(prediction_now==y_t)/size(y_t,1);
%                 acc=acc+acc_temp;
%             elseif t>=2
%                 %对数据进行增广，利用上一轮的模型
%                 prediction_pre=[];
%                 if full==0
%                     for model_num=1:buffer_index-1
%                         pred_temp=predict(buffer{buffer_index},X_t);  
%                         prediction_pre=[prediction_pre,pred_temp];
%                     end
%                 elseif full==1
%                     model_index=1:P+1;
%                     model_index=model_index(find(model_index~=buffer_index));
%                     for model_num=model_index
%                         pred_temp=predict(buffer{buffer_index},X_t);  
%                         prediction_pre=[prediction_pre,pred_temp];
%                     end
%                 end
%                 XA_t=[X_t,prediction_pre];
%                 %利用上一轮的模型进行预测
%                 prediction_now=predict(model_previous,XA_t);
%                 acc_temp=sum(prediction_now==y_t)/size(y_t,1);
%                 acc=acc+acc_temp;
%             end
%-----------学习增广后的本轮模型---------------------   


            if t==0
                model_recent=fitcsvm(X_t,y_t);
            else
                prediction=[];
                if full==0     %a little bug
                    if buffer_index==P+1
                        for model_num=2:buffer_index
                            pred_temp=predict(buffer{buffer_index},X_t);  
                            prediction=[prediction,pred_temp];
                        end
                    else
                        for model_num=1:buffer_index
                            pred_temp=predict(buffer{buffer_index},X_t);  
                            prediction=[prediction,pred_temp];
                        end
                    end
                    
                elseif full==1
                    model_index=1:P+1;       
                    temp=buffer_index+1;
                    if temp>P+1
                        temp=1;
                    end
                    model_index=model_index(find(model_index~=temp));
                    for model_num=model_index
                        pred_temp=predict(buffer{buffer_index},X_t);  
                        prediction=[prediction,pred_temp];
                    end
                end
                model_recent=fitcsvm([X_t,prediction],y_t);
            end
            model_previous=model_recent;
            

%在测试集上进行预测：

            if t==0
                model_recent=fitcsvm(X_t,y_t);
                prediction_t=predict(model_recent,Xt_test);
                acc_temp=sum(prediction_t==yt_test)/chunkSize;
            else
                prediction=[];
                prediction_label_test=[];
                if full==0     %a little bug
                    if buffer_index==P+1
                        for model_num=2:buffer_index
                            pred_temp=predict(buffer{buffer_index},X_t); 
                            pred_test=predict(buffer{buffer_index},Xt_test);
                            prediction=[prediction,pred_temp];
                            prediction_label_test=[prediction_label_test,pred_test];
                        end
                    else
                        for model_num=1:buffer_index
                            pred_temp=predict(buffer{buffer_index},X_t);
                            pred_test=predict(buffer{buffer_index},Xt_test);
                            prediction=[prediction,pred_temp];
                            prediction_label_test=[prediction_label_test,pred_test];
                        end
                    end
                    
                elseif full==1
                    model_index=1:P+1;       
                    temp=buffer_index+1;
                    if temp>P+1
                        temp=1;
                    end
                    model_index=model_index(find(model_index~=temp));
                    for model_num=model_index
                        pred_temp=predict(buffer{buffer_index},X_t);  
                        pred_test=predict(buffer{buffer_index},Xt_test);
                        prediction=[prediction,pred_temp];
                        prediction_label_test=[prediction_label_test,pred_test];
                    end
                end
                model_recent=fitcsvm([X_t,prediction],y_t);
                XAt_test=[Xt_test,prediction_label_test];
                prediction_t=predict(model_recent,XAt_test);             
                acc_temp=sum(prediction_t==yt_test)/chunkSize;
            end
            Plot_recorder(i,t)=acc_temp;

%-----------利用本轮数据训练模型并存储模型----------------
            model_buffer=fitcsvm(X_t,y_t);
            buffer_index=buffer_index+1;
            if(buffer_index<=P+1)
                buffer{buffer_index}=model_buffer;
            elseif(buffer_index==P+2)
                buffer_index=1;
                buffer{buffer_index}=model_buffer;
                full=1;
            end            
        end    
 %   end
    save(['./Plot_Result_TIX/plot',num2str(i)],'Plot_recorder')
    i
end
