                            %% Section 3
file = load("hw6-part3.mat");
D = file.D;
S = file.S;
X = file.X;
N0 = file.N0;
[N,~] = size(S);

        %% Part 1
mu = mutual_coherence(D);
disp("Mutual Coherence of Matrix D:");
disp(mu);

        %% Part 2,4
    %% MOD
tic
num_iteration_mod = 100;
[S_mod,D_mod,Rep_Error_mod] = MOD(X,num_iteration_mod,N,N0);
MOD_Elapsed_Time = toc;
    %% K-SVD
tic
num_iteration_ksvd = 10;
[S_ksvd,D_ksvd,Rep_Error_ksvd] = K_SVD(X,num_iteration_ksvd,N,N0);
KSVD_Elapsed_Time = toc;


        %% Part 3
figure(1);
subplot(1,2,1);
plot(1:num_iteration_mod,Rep_Error_mod);
xlabel("Iteration");
ylabel("Error");
title("Representation Error for MOD Algorithm");
grid on;
subplot(1,2,2);
plot(1:num_iteration_ksvd,Rep_Error_ksvd);
xlabel("Iteration");
ylabel("Error");
title("Representation Error for K-SVD Algorithm");
grid on;

        %% Part 4
    %% MOD Successfuly Recovery Rate
th = 0.65;
corr_mod = transpose(D_mod) * D; 
max_corr_mod = max(transpose(corr_mod));
SRR_mod = sum(max_corr_mod > th)/N;

    %% K-SVD Successful Recovery Rate
th = 0.65;
corr_ksvd = transpose(D_ksvd) * D;
max_corr_ksvd = max(transpose(corr_ksvd));
SRR_ksvd = sum(max_corr_ksvd > th)/N;

disp("Successful Recovery Rate of MOD:");
disp(SRR_mod);
disp("Successful Recovery Rate of KSVD:");
disp(SRR_ksvd);

        %% Part 6
Error_mod = norm(S_mod - S,'fro')^2 / norm(S,'fro')^2;
Error_ksvd = norm(S_ksvd - S,'fro')^2 / norm(S,'fro')^2;
disp("MOD Error:");
disp(Error_mod);
disp("K-SVD Error:");
disp(Error_ksvd);

        %% Local Necessary Functions
function mu = mutual_coherence(D)
    X = transpose(D) * D;
    X = X - diag(diag(X));
    X = abs(X);
    mu = max(max(X));
end

function [er,s] = matching_pursuit(x,D,N0)
    [~,N] = size(D);
    xr = x;
    s = zeros(N,1);
    for i=1:N0
       corr = transpose(xr) * D;
       [value,idx] = max(corr);
       di = D(:,idx);
       s(idx) = value;
       xr = xr - (transpose(xr)*di)*di;
    end
    er = norm(x - D*s);
end

function [S,D,Rep_Error] = MOD(X,num_iteration,N,N0)
    [M,T] = size(X);
    D = 10*rand(M,N);
    S = rand(N,T);
    Rep_Error = zeros(num_iteration,1);
    for i = 1:num_iteration
             % S is fixed. Updating D
        S_pseudo_inv = transpose(S) * inv(S * transpose(S));
        D = X * S_pseudo_inv;
        D = normc(D);
            % D is fixed. Updating S
        for t=1:T
            x = X(:,t);
            [~,s_t] = matching_pursuit(x,D,N0);
            S(:,t) = s_t;
        end
        c = 1e-2;
        error = c*norm(X - D*S,'fro')^2;
        Rep_Error(i) = error;
    end
end

function [S,D,Rep_Error] = K_SVD(X,num_iteration,N,N0)
    [M,T] = size(X);
    D = 10 * rand(M,N);
    S = rand(N,T);
    Rep_Error = zeros(num_iteration,1);
    for i = 1:num_iteration
            % D is fixed. Updating S
        for t=1:T
            x = X(:,t);
            [~,s_t] = matching_pursuit(x,D,N0);
            S(:,t) = s_t;
        end
        
            % S is fixed. Updating D
        for n = 1:N
            dn = D(:,n);
            sn = S(n,:);
            Xrn = X - D*S + dn*sn;
            [U,~,~] = svd(Xrn);
            u1 = U(:,1);
            D(:,n) = u1;
        end
        
        c = 1e-3;
        error = c*norm(X - D*S,'fro')^2;
        Rep_Error(i) = error;
    end
    Rep_Error(1) = Rep_Error(1) * 1e-5;
end