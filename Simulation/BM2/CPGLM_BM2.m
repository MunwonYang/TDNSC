% Setup Cluster for Parallel computing in Matlab
n_cores = 26;
cluster = parpool(n_cores);

% Model Setting for BM2
p1 = 64;
p2 = 64;
p = p1*p2;

n1 = 75;
n2 = 75;
n = n1+n2;
prob1 = n1/n;
prob2 = n2/n;

Y = [repmat(1,[n1,1]),repmat(2,[n2,1])];
Y = Y(:);
K = 2;
dimen = [p1,p2];

model = 'binomial'; % CP-glm(logistic regression)
pentype = 'enet'; % Penalty function type
penparam = 1; % Lasso Penalty

% Setting each class centroid
v1 = zeros(p1,p2);
v2 = zeros(p1,p2);
v1([1,2,11,12],[1,2]) = 0.25;
v1([1,2,11,12],[11,12]) = 0.4;
v2([1,2,11,12],[1,2]) = -0.25;
v2([1,2,11,12],[11,12]) = -0.4;

% Setting covariance matrix
sigma1 = eye(p1);
sigma2 = eye(p2);

for i = 1:p1
    for j = 1:p1
        sigma1(i,j) = 0.8^abs(i-j);
    end
end

for i = 1:p1
    for j = 1:p1
        if i ~= j
            sigma2(i,j) = 0.5;
        end
    end
end

V = {v1,v2};
sigma = {sigma1,sigma2};

% Calculating each class tensor mean
mu = {};
for i = 1:K
    mu{i} = sqrtm(sigma{1}) * V{i} * sqrtm(sigma{2});
end

dsigma = {};
for i = 1:numel(dimen)
    dsigma{i} = sqrtm(sigma{i});
end

nfold = 5; % Number of fold
Run = 100; % Number of replicates
lam = zeros([Run,1]); % Storage of optimal lambda value
err = zeros([Run,1]); % Storage of each simulation error
Ran = zeros([Run,1]); % Storage of optimal rank R

tic;
parfor (run = 1:Run,cluster)
    % Fix the random seed for each simulation
    rng(1996*run);
    trsize = n;
    % Generating train tensor normal random variable
    X = zeros([dimen,trsize]);
    for i = 1:n
        Z = tensor(randn(dimen));
        X(:,:,i) = ttm(Z,dsigma,[1,2]) + mu{Y(i)};
    end

    Rank = [1,2,3,4]; % CP rank(1,2,3,4)
    lambda=[0.001,0.005,0.01,0.1,0.3,0.5,1,2,5,7.5,10,15]; % Candidate lambda value
    error = zeros(length(Rank), length(lambda));

    for i = 1:length(Rank)
        for j = 1:length(lambda)
            cv = zeros([nfold,1]);
            % 5 fold Cross-validation
            for k =1:nfold
                [Train,Test,Y_Train,Y_Test] = fold_5(X,Y,dimen,k);
                % Rough estimate from reduced sized data
                xsmall = array_resize(Train, [24,24,size(Train,3)]);
                Z = ones(length(Y_Train),1);
                [~,beta_small]=kruskal_sparsereg(Z,xsmall,Y_Train-1,Rank(i),model,lambda(j),pentype,penparam);
                % warm start from coarsened estimate
                [beta0,beta_rk1,~,~]=kruskal_sparsereg(Z,Train,Y_Train-1,Rank(i),model,lambda(j),pentype,penparam,'B0',array_resize(beta_small,dimen));
                % Estimated parameter for CP-GLM(Logistic Regression)
                escoef=khatrirao(beta_rk1.U{2},beta_rk1.U{1})*ones(Rank(i),1);

                % Estimation of binary class label
                pred = zeros([length(Y_Test),1]);
                for l = 1:length(Y_Test)
                    pred(l) = sum(escoef .* reshape(Test(:,:,l),[prod(dimen),1])) + beta0;
                end

                pred(pred >= 0) = 2;
                pred(pred <0) = 1;
                cv(k) = sum(pred ~= Y_Test) / length(Y_Test) * 100;
            end
            error(i,j) = mean(cv);
        end
    end
    error1 = reshape(error, [(length(Rank) * length(lambda)),1]);
    ind  = find(min(error1) == error1);
    ind = min(ind);

    % Choose optimal lambda value through 5 fold cross validation
    C = ceil(ind/length(Rank));
    
    % Choose optimal rank through 5 fold cross validation
    if rem(ind,length(Rank)) == 0
        R = length(Rank);
    else
        R =  rem(ind,length(Rank));
    end

    % Generate test tensor normal random variable
    X1 = zeros([dimen,n]);
    for i = 1:n
        Z = tensor(randn(dimen));
        X1(:,:,i) =  ttm(Z,dsigma,[1,2]) + mu{Y(i)};
    end 
    
    % Rough estimate from reduced sized data
    Xsmall = array_resize(X, [24,24,size(X,3)]);
    Z = ones(length(Y),1);
    [~,beta_small]=kruskal_sparsereg(Z,Xsmall,Y-1,Rank(R),model,lambda(C),pentype,penparam);
    % warm start from coarsened estimate
    [beta0,beta_rk1,~,~]=kruskal_sparsereg(Z,X,Y-1,Rank(R),model,lambda(C),pentype,penparam,'B0',array_resize(beta_small,dimen));
    % Estimated parameter for CP-GLM(Logistic Regression)
    % Rank and lambda are given by 5 fold Cross-validation
    escoef=khatrirao(beta_rk1.U{2},beta_rk1.U{1})*ones(Rank(R),1);

    % Estimation of binary class label
    pred = zeros(length(Y),1);

    for i = 1:n
    pred(i) = sum(escoef .* reshape(X1(:,:,i),[prod(dimen),1])) + beta0;
    end

    pred(pred >= 0) = 2;
    pred(pred <0) = 1;

    err(run) = sum(Y ~= pred) / length(Y) * 100;
    lam(run) = lambda(C); % Optimal lambda value for each simualtion 
    Ran(run) = Rank(R); % Optimal rank for each simualtion
end
toc;

mean(err) % Average classification error rate for 100 simulations
std(err)/sqrt(Run) % Standard error rate on 100 simualtions
tabulate(lam) % Distribution of lambda value for 100 simulations
tabulate(Ran) % Distribution of Rank for 100 simulations


% Close down the cluster
delete(cluster);

% Function for 5-fold cross validation
function [Train,Test,Y_Train,Y_Test] = fold_5(X,Y,sz,p)
n = length(Y);
idx = randperm(n);
P = prod(sz);
m = idx(round((p-1)/5*n)+1:round(p*n/5));
A = round(0.8*n);
T = zeros([n,P+1]);
for i = 1:n
    T(i,:) = [reshape(X(:,:,i),[1,P]), Y(i)];
end

test = T(m,:);
train = T(setdiff(idx,m),:);

Train = zeros([sz,A]);
Test = zeros([sz,n-A]);

for i = 1:A
    Train(:,:,i) = reshape(train(i,1:end-1),sz);
end

for i = 1:n-A
    Test(:,:,i)  = reshape(test(i,1:end-1),sz);
end

Y_Train = train(:,end);
Y_Test = test(:,end);
end
