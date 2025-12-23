% Setup cluster for parallel computing in Matlab
n_cores = 30;
cluster = parpool(n_cores);

p1 = 6; % Six pre-specified sites for each tooth by hygienists
p2 = 28; % Total number of tooth
p3 = 2; % Measurement of periodontal pocket depth(PPD), and clinical attachment level(CAL)
dimen = [p1,p2,p3]; % Dimension of dataset
p = prod([p1,p2,p3]); % Number of variables
q = 5; % Five covariates(Age, gender, body mass index(BMI), smoking status, hemoglobin A1c(HbAlc))
n = 153; % Number of data

model = 'binomial'; % CP-glm(logistic regression)
pentype = 'enet'; % Penalty function type
penparam = 1; % Lasso Penalty

% Recall the PD data
myy = readtable("myy.csv");
myy = myy(:,2:337);
myx = readtable("myx.csv");
myx = myx(:,2:6); 

Y = table2array(myx(:,2)); % Set class label(Gender: Male, Female)
myy = table2array(myy);
X = zeros([6,28,2,n]); % Set tensor predictor(6*28*2*n)
myx = table2array(myx);

Run = 100; % Number of replicates
err = zeros([Run,1]); % Storage of each simulation error

parfor (run = 1:Run,cluster)
    % Fix the random seed for each simulation
    rng(1996*run);
    % Split randomly. Train data(65%), and Test data(35%)
    hpartition = cvpartition(n,'Holdout',0.35);
    idxTrain =training(hpartition);
    idxTest = test(hpartition);

    Train = myy(idxTrain,:); % Tensor predictor for Train data
    Y_Train = Y(idxTrain); % Class label for Train data
    Test = myy(idxTest,:); % Tensor predictor for Test data
    Y_Test = Y(idxTest); % Class label for Test data

    Xtrain = zeros([p1,p2,p3,length(Y_Train)]);
    Xtest = zeros([p1,p2,p3,length(Y_Test)]);

    for i = 1:length(Y_Train)
        Xtrain(:,:,:,i) = reshape(Train(i,:),[p1,p2,p3]);
    end

    for i = 1:length(Y_Test)
        Xtest(:,:,:,i) = reshape(Test(i,:),[p1,p2,p3]);
    end

    Rank = [2,3,4]; % CP rank(2,3,4)
    lambda=[0.001,0.01,0.1,0.5,1,5,10,15,30, 50]; % Candidate Lambda value
    error = zeros(length(Rank), length(lambda));

    for i = 1:length(Rank)
        for j = 1:length(lambda)
            % 5 fold Cross-validation
            cv = cvpartition(length(Y_Train), "KFold", 5);
            cv_error = zeros([5,1]);
            for fold = 1:5
                tind = training(cv,fold);
                find = test(cv,fold);
                Xtr = Xtrain(:,:,:,tind);
                Ytr = Y_Train(tind);
                Xv = Xtrain(:,:,:,find);
                Yv = Y_Train(find);
                % Rough estimate from reduced sized data 
                Xsmall = array_resize(Xtr,[3,10,2,size(Xtr,4)]);
                Z = ones(length(Ytr),1);
                [~,beta_small]=kruskal_sparsereg(Z,Xsmall,Ytr,Rank(i),model,lambda(j),pentype,penparam);
                % warm start from coarsened estimate
                [beta0,beta_rk1,~,~]=kruskal_sparsereg(Z,Xtr,Ytr,Rank(i),model,lambda(j),pentype,penparam,'B0',array_resize(beta_small,dimen));
                % Estimated parameter for CP-GLM(Logistic Regression)
                escoef=khatrirao(beta_rk1.U{3},beta_rk1.U{2},beta_rk1.U{1})*ones(Rank(i),1);

                % Estimation of binary class label
                pred = zeros([length(Yv),1]);
                for l = 1:length(Yv)
                    pred(l) = sum(escoef .* reshape(Xv(:,:,:,l),[p,1])) + beta0;
                end

                pred(pred >= 0) = 1;
                pred(pred < 0) = 0;
                cv_error(fold) = sum(pred ~= Yv) / length(Yv) * 100;
            end
            error(i,j) = mean(cv_error);
        end
    end

    % Choose optimal lambda value through 5 fold cross validation
    C = ceil(ind/length(Rank));
    
    % Choose optimal rank through 5 fold cross validation
    if rem(ind,length(Rank)) == 0
        R = length(Rank);
    else
        R =  rem(ind,length(Rank));
    end

    % Generate test tensor normal random variable
    Xsmall1 = array_resize(Xtrain,[3,10,2,size(Xtrain,4)]);
    Z = ones(length(Y_Train),1);
    [~,beta_small]=kruskal_sparsereg(Z,Xsmall1,Y_Train,Rank(R),model,lambda(C),pentype,penparam);
    % warm start from coarsened estimate
    [beta0,beta_rk1,~,~]=kruskal_sparsereg(Z,Xtrain,Y_Train,Rank(R),model,lambda(C),pentype,penparam,'B0',array_resize(beta_small,dimen));
    % Rank and lambda are given by 5 fold Cross-validation
    escoef=khatrirao(beta_rk1.U{3},beta_rk1.U{2},beta_rk1.U{1})*ones(Rank(R),1);

    % Estimation of binary class label
    pred = zeros(length(Y_Test),1);

    for i = 1:length(Y_Test)
        pred(i) = sum(escoef .* reshape(Xtest(:,:,:,i),[prod(dimen),1])) + beta0;
    end

    pred(pred >= 0) = 1;
    pred(pred <0) = 0;


    err(run) = sum(Y_Test ~= pred) / length(Y_Test)*100; % Error rate for each simulation
end

mean(err) % Average classification error rate for 100 simulations
std(err) / sqrt(Run) % Standard error rate on 100 simualtions

% Close down the cluster
delete(cluster);