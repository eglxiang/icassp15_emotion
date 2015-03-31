% 2014-04-03 Xiang Xiang xxiang@cs.jhu.edu
% This programs implements
% Eigenface with S nearest neighbors
% Eigenface with nearest subspace
% Sparse Representation based Classification
% labels can start from any number, but they need to be continuous 

clc
clear all;
close all;
addpath('./Recon');

%% parameters you set!
runnum = 20; % num of runs (trails)
ifresize = 0; % do image downsampling?
    if ifresize
        dratio = 0.25; % set to 1 if no downsampling
        rfaces = permute(faces, [2 3 1]);  % rearrange dimensions
        drfaces = imresize(rfaces, dratio); % downsample each image.
        faces = permute(drfaces, [3,1,2]); % rearrange dim back to original
    end
useeig = 0; % use eigenface? true/false
if useeig
    rho = 0.005; % top k percent eigenvectors will be retained.
    usenn = 0; % use eigen with kNN?
    if usenn == 1    
        S = 10; % S-Nearesr Neighbor;
        global_ccrate_1nn=0;
        global_ccrate_Snn=0;
        global_confu_nn =0;
    end
    usens = 1; % use eigne with Neareast Subspace?
    if usens == 1
        global_ccrate_ns=0;
        global_confu_ns = 0;
    end
end
usesrc = 1; % use SRC?
if usesrc
    srceq = 0; % for SRC, use equality constraint? 1-yes, eq; 0-no, ineq
    if srceq
        optm_alg = 'LINPROG'; %LINPROG works with srceq=1
    else
        optm_alg = 'FISTA'; % ALM,OMP,FISTA,GPSR,SP;
        if strcmp(optm_alg,'OMP') || strcmp(optm_alg, 'SP')
            sparsity = 60; % sparsity chosen out of number of trn samples
        end
        if strcmp(optm_alg,'GPSR')
            lambda = 0.05; % regularization parameter for tradeoff btw data fitting and l1 norm
        end
    end
    global_ccrate_src=0;
    global_confu_src = 0;
end
load('emotions_faces.mat');
load('neutral_faces.mat');
    faces = double(emotions_faces - neutral_faces);
    [P,H,W] = size(faces); % num of images, height, width
    D = H*W; % image dim
load('labels.mat');
    facecls = labels;
    
for t=1:runnum % trial num
    %% assign training and testing samples
    %-- not fully random partition but sort of, for we don't know the order
    % R = ceil(P/2); %trnnum
    % E = P - R; %tstnum
    % trn = zeros(R,1);
    % tst = zeros(E,1);
    % j = 0;
    % if mod(P,2) == 0
    %     for i=1:2:P
    %         j = j+1;
    %         trn(j) = i;
    %         tst(j) = (i+1);
    %     end
    % else
    %     for i=1:2:(P-1)
    %         j = j+1;
    %         trn(j) = i;
    %         tst(j) = i+1;
    %     end
    %     trn(j+1) = i+2;
    % end

    %-- real random partition in each class
    prev_label=1;
    prev_start_id=1;
    count_trn=0;
    count_tst=0;
    trn=zeros((floor(P/2)-10),1);
    tst=zeros((floor(P/2)-10),1);
    facecls_ex = facecls; % extended
    facecls_ex(P+1)=0; % augment 1 label to avoid lossing the last subject
    %C = max(facecls) - min(facecls) + 1; % assume continuous labels
    C = 0; % number of classes
    num_per_cls = zeros(2,1); % assume at least 2 classes
    num_per_trn = zeros(2,1);
    num_per_tst = zeros(2,1);
    for i=1:(P+1) % augmented one label
        if facecls_ex(i) ~= prev_label
            C = C + 1;
            oneclass = prev_start_id:(i-1);
            num = i-prev_start_id;
            num_per_cls(C) = num;
            num_per_trn(C) = 0; % count
            num_per_tst(C) = 0; % count

            raw_id = randperm(num,num);

            onecls_id = raw_id + (prev_start_id-1);
            for j=1:ceil(num/2)
                count_trn = count_trn + 1;
                num_per_trn(C) = num_per_trn(C) + 1;
                trn(count_trn) = onecls_id(j);
            end
            for j=(ceil(num/2)+1):num
                count_tst = count_tst + 1;
                num_per_tst(C) = num_per_tst(C) + 1;
                tst(count_tst) = onecls_id(j);
            end
            prev_label = facecls_ex(i);
            prev_start_id = i;
        end
    end
    R = length(trn); %num of tRaining samples
    E = P - R; %num of tEsting samples

    %% --eigenface based method
    if useeig
        %--- training
        meanface3d = mean(faces(trn,:,:));
        meanface = squeeze(meanface3d(1,:,:)); % to 2d.
        % figure, imshow(meanface, [0 255]); % show the meanface
        vecmsfaces = reshape((faces(trn,:,:)-repmat(meanface3d, [R 1])), [R,D]); % mean-shifting and vectorization, PCA needs to process 2d array
        [evec,score,eval] = pca(vecmsfaces); % already sorted, pca: 'Economy' in default
        % evec:DxR; score:Rx(R-1); eval:Rx1
        pevec = evec(:,1:floor(rho*D)); % DxT, principal eigenvectors, pick top eigenvectors
        %figure, imshow(reshape(pevec(:,1), [H W]), [0 255]); % show the top eigenface
        rep_trn = pevec' * vecmsfaces'; % TxR, project faces into principal eigenspace to form representation for trn

        %--- testing (by looping)
        rep_tst = zeros(floor(rho*D),E);
        %% Neareast Neighbor
        if usenn
            hit_nn = 0;
            count_mat_nn = zeros(C,C);
            for i=1:E
                %-- compute representation for tst, for both NN and NS
                curtst = squeeze(faces(tst(i),:,:)); % current test image
                diff = curtst - meanface; % H x W
                diff = diff(:);  % vectorize to D x 1
                % project testing to subspace (eigenspace)
                rep = pevec' * diff; % Tx1
                rep_tst(:,i) = rep;
                %-- similarity [0,inf) of the test to each training image
                simscore_nn = arrayfun( @(n) (1/(1+norm(rep - rep_trn(:,n)))), 1:R );          
                % 1-NN for confusion: find the highest similarity (problem: multi topest)
                [max_score, max_idx] = max(simscore_nn);
                count_mat_nn(facecls(tst(i))-min(facecls)+1, facecls(trn(max_idx))-min(facecls)+1) = ...
                    count_mat_nn(facecls(tst(i))-min(facecls)+1, facecls(trn(max_idx))-min(facecls)+1) + 1;
                % recognition rate by S-NN (more tolerant than 1NN, which is too strict)
                [sort_score, sort_idx] = sort(simscore_nn, 'descend');
                % candidate pool of identity
                [~,~,cand]= mode( facecls(trn(sort_idx(1:S))) );
                cand = cand{1,1};
                if any(cand == facecls(tst(i)))
                    hit_nn = hit_nn + 1;
                end
            end
            % confusion matrix
            confu_mat_nn = zeros(C, C);
            for i=1:C
                confu_mat_nn(i,:) = count_mat_nn(i,:) / num_per_tst(i);
            end
            ccrate_1nn = sum(diag(count_mat_nn)) / E;
            ccrate_Snn = hit_nn / E; % Evaluation, correct classfication rate by S-NN
            global_confu_nn = global_confu_nn + confu_mat_nn;
            global_ccrate_1nn = global_ccrate_1nn + ccrate_1nn;
            global_ccrate_Snn = global_ccrate_Snn + ccrate_Snn;
        end

        %% Nearest Subspace
        if usens
            count_mat_ns = zeros(C,C);
            for i=1:E
               %-- compute representation for tst, for both NN and NS
               curtst = squeeze(faces(tst(i),:,:)); % current test image
               diff = curtst - meanface; % H x W
               diff = diff(:);  % vectorize to D x 1
               % project testing to subspace (eigenspace)
               rep = pevec' * diff; % Tx1
               rep_tst(:,i) = rep;
               %-- similarity [0,inf) of the test to each training subspace
               simscore_sub = zeros(1,C); % each class of training samples form a subspace
               for j=1:C
                   % which trn are in one class
                   onecls_ns = find(facecls(trn) == (min(facecls)+j-1)); % Mx1
                   M = length(onecls_ns); % class size
                   if M ~= num_per_trn(j)
                       disp('Num of training samples does not match! '); break;
                   end
                   trncls = reshape(faces(trn(onecls_ns), :,:), [M, D]);
                   trncls = trncls'; % DxM
                   alpha = trncls \ curtst(:); % Mx1, project testing to the subspace, gives the coeffs
                   % similarity of the testing to each training class (subspace)
                   simscore_sub(j) = 1/(1+norm(rep_tst(:,i)-rep_trn(:,onecls_ns)*alpha));
               end % note that this may be biased (imblanced training data, class 2 has too few training samples)
               % which class is the most similiar (1-nearest-subspace)
               [max_score_sub, max_idx_sub] = max(simscore_sub);
               % confusion matrix
               count_mat_ns(facecls(tst(i))-min(facecls)+1, max_idx_sub) = ...
                    count_mat_ns(facecls(tst(i))-min(facecls)+1, max_idx_sub) + 1;
            end
            confu_mat_ns = zeros(C, C); % confusion matrix
            for i=1:C
                confu_mat_ns(i,:) = count_mat_ns(i,:) / num_per_tst(i);
            end
            ccrate_ns = sum(diag(count_mat_ns)) / E;
            global_confu_ns = global_confu_ns + confu_mat_ns;
            global_ccrate_ns = global_ccrate_ns + ccrate_ns;
        end
    end    

    %% Sparse Representation based Classification
    if usesrc
        %-- step 1: form the dictionary
        dict = reshape(faces(trn,:,:),[R,D]);
        dict = dict'; % DxR
        %-- step 2: normalize each columan
        for i=1:R
            dict(:,i) = dict(:,i) ./ norm(dict(:,i));
        end    
        %-- step 3: solve the l1 min problem
        count_mat_src = zeros(C,C);    
        for i=1:E % E optm problems
            curtst = squeeze(faces(tst(i),:,:)); % current test image
            curtst = curtst(:); % Dx1
            % recovery coefficient vector coef for the 
            if srceq == 1 % equality constraint
                B = [dict, -dict]; %Dx2R, prepared for lingprog
                f = ones(2*R,1); % 2Rx1
                lb = zeros(2*R,1); % z >= 0
                %-- l1-min by linear prog, which suffers from inconsistent equality constraint.
                z = linprog(f,[],[],B,curtst,lb); % 2Rx1
                zlen = size(z, 1);
                coef = z(1:(zlen/2), :) - z((zlen/2 + 1):zlen, :); % get coef: Rx1
            else % inequality constraint 
                %epsilon = 10^(-5); % error tolerance
                switch optm_alg
                    case 'ALM'
                        [coef,~] = ALM(dict, curtst);
                    case 'OMP'
                        coef = OMP(curtst, dict, sparsity); % Rx1. sparsity chosen out of number of trn samples
                    case 'FISTA'
                        [coef,~] = FISTA(dict,curtst);
                    case 'GPSR'
                        [coef,~,~,~,~,~] = GPSR_Basic(curtst,dict,lambda);
                    case 'SP'
                        coef = SP(curtst, dict, sparsity);
                end    
            end        
            %-- step 4: compute residue to each class
            res = zeros(C,1);
            for j = 1:C
               % which trn are in one class % found it, that's a bug!!!
               oneclscoef = zeros(R,1); % Rx1
               onecls_src = find(facecls(trn) == (min(facecls)+j-1)); % Mx1, index for trn
               oneclscoef(onecls_src) = coef(onecls_src); % find coef for the training samples in one class
               % residue 
               res(j) = norm(curtst - dict*oneclscoef); % residue, reconstruction error, l2 norm in default
            end
            [match_score_src, match_idx_src] = min(res); % which class?
            % confusion, assume continuous label
            count_mat_src((facecls(tst(i))-min(facecls)+1),match_idx_src) = ...
                count_mat_src((facecls(tst(i))-min(facecls)+1),match_idx_src) + 1;       
        end
        % confusion matrix
        confu_mat_src = zeros(C, C);
        for i=1:C
            confu_mat_src(i,:) = count_mat_src(i,:) / num_per_tst(i);
        end
        ccrate_src = sum(diag(count_mat_src)) / E;
        global_confu_src = global_confu_src + confu_mat_src;
        global_ccrate_src = global_ccrate_src + ccrate_src;
    end
end

if useeig
    if usenn
        global_ccrate_1nn = global_ccrate_1nn/runnum %eig-NN
        global_ccrate_Snn = global_ccrate_Snn/runnum
        global_confu_nn = global_confu_nn/runnum
    end
    if usens
        global_ccrate_ns = global_ccrate_ns/runnum %eig-NS
        global_confu_ns = global_confu_ns/runnum
    end
end
if usesrc
    global_ccrate_src = global_ccrate_src/runnum % SRC
    global_confu_src = global_confu_src/runnum
end