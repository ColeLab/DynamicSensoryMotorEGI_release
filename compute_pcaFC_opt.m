function [output]=compute_pcaFC_opt(data_input,use_PCA,opt_mode,target_regions,model_order,zscore_tseries,include_contemp,include_autoreg,outfile,cv_max,cv_increment,cv_run,sub_id,out_dir)
 
%Author: Ravi Mill, rdm146@rutgers.edu
%Last update: Jan 15th 2021

%DESCRIPTION: function that computes MVAR FC (with PCA optimization) for rest data in Dynamic Activity
%Flow project: Mill et al, "Causal emergence of task information from dynamic 
%network interactions in the human brain"

%Note: Type of MVAR FC computed depends on 'use_PCA', with 2 versions possible based on 
%the number of PCs (nPCs) used in each regression of source/autoreg terms-> targets:
%use_PCA=3: Optimization of nPCs via blocked crossval (RECOMMENDED): 
    %next trial MSE is minimized (i.e. train on all tpoints in pseudotrial one, 
    %test on all tpoints in trial2; train on all tpoints in trial 2, test on 3
    %etc for all test trial loops for each PC); this preserves temporal ordering of timeseries data (i.e. by not
    %training on timepoints that emerge after the test timepoints) and is more
    %suited to our desired 'future prediction' via activity flow.
    %Next-trial testing is also recommended given direct scaling of optimisation metric (target MSE) with
    %numPCs (i.e. optimal PCs=maxPC) if testing on tpoints within the same trial.
    %MSE for each PC=average across targets x trial loops; opt nPC selected basedd on minimum MSE
    %across all PCs
%use_PCA>3:A priori PC selection (only recommended if blocked cv will take too long): 
%integer for number of PCs to retain e.g. use_PCA=100 will run regressions
%using first 100 PCs

%Full conceptual explanation of blocked crossval (use_PCA=3):
%1. Firstly set the range of PCs to search over (can loop in increments if the range is too large).
%2. Then loop through targets (Y) and each trial/testing fold: run the PCA on the predictors (X; source lagged terms + target autoregs), to give PCAScoresX
%3. Then loop through regressions over the range of PCs, varying nPC:
%regress PCAScoresX (1:nPC) -> Y, with next-trial crossvalidation (fold 1: train on
%all tpoints in trial1, test on all tpoints in trial2; fold 2: train on all
%tpoints in trial 2, test on all tpoints in trial 3 etc)
%4. Transform the PCAbetas back to the original space
%5. Use the resulting betas to predict the test timeseries (for the next
%trial; i.e. dot of Xtest activations with the trained betas, to yield Yhat
%i.e. target activation predictions for the next trial
%6. Compute the mean squared error (MSE) between Y and the Yhat predictions
%7. After looping through all trial folds, you will end up with MSE dimensions=number of PCs x target regions x num trials
%8. Average MSE over target regions and trial folds, to get MSE for that
%nPC value
%9. Loop through all nPCs in the specified range, and choose the optimized
%nPC for that subject as that which yields the minimum mean MSE
%10. Use that optimal nPC value in the final regressions estimating betas
%(FC weights) for each target region; resulting MVAR FC matrix is used to
%compute activity flow for that subject
%*General justification for use of optimization vs maxPCs: maxPCs will
%overfit to noise in the full timeseries, as some PCs will reflect noise
%rather than signal; testing on subsets of the timeseries maximises fit of the model fo the signal of interest, 
%rather than noise in the data 
%*Justification for blocked vs standard crossval - see explanation of use_PCA=3 above

%**v4 of script=the result of PC optimization (use_PCA=3) is written for each npc *separately*, so that if there are any job abort events
%(which is happening randomly on the OARC cluster) then you only lose ~35 hours of comp time
%(versus 2 weeks). 
%The script hence runs in two modes (determined by opt_mode), see explanation below.

%INPUTS:
%data_input: cell containing input tseries data for a subject, dimensions: 1 x num_trials,
%   each cell contains a num_regions x num_tp (timepoints) array; 
%   note that at present each trial cell must have same num_regions and num_tp
%use_PCA: numeric, determines number of PCs entered into 
%   regression predicting target tseries vector from source tseries matrix;
%   *see above for description of different methods of doing this
%opt_mode: numeric binary, sets which mode is run by the function:
%   1=run optimization for specific PC values (cv_run), writing these out for
%       each pc (step1); ONLY applies to blocked cv i.e. is use_PCA=3
%   2=collate results of optimization of specific PC values (from step1), find
%       the optimal nPC and estimate the final MVAR FC based on it (step2);
%       OR run for a priori value of nPCs i.e. use_PCA=100
%   *For blocked CV (use_PCA=3), only run step2 after step1;
%   for a priori nPCs (e.g. use_PCA=100), only run step2
%target_regions: numeric row vector (1xn), indexes which regions in each trial in
%   data_input are the 'target' to-be-predicted regions
%model_order: numeric; basis of MVAR model: how many lagged tpoints to include in source predictor
%   tseries (and target autoregs, if specified) predicting each target tpoint (t0), 
%   e.g. if set to 10 will include lagged tpoints from t0-1 to t0-10
%zscore_tseries: numeric binary, if set to 1 (recommended) will zscore each trial's tpoints for
%   each region separately prior to estimating FC
%include_contemp: numeric binary, if set to 1 (recommended) will include contemporaneous
%   (t0) terms for all source regions in the MVAR model; this is recommended to improve MVAR model 
%   estimation of the lagged terms (even if the t0 terms are excluded from subsquent actflow step)
%include_autoreg: numeric binary, if set to 1 (recommended) will include lagged (self-coupling) terms for
%   the target region being predicted in the MVAR model i.e. autoregressive terms 
%   (t0-1 -> model_order); again this is recommended to improve accuracy of
%   the source lagged terms
%outfile: string, full path to where output will be saved (containing optimization results and
%   final MVAR FC matrices; see outputs below for details)
%cv_max: numeric, sets max PC to iterate over for optimization (if use_PCA=3, otherwise leave empty)
%cv_increment: sets increments of PCs to iterate over during optimization (if use_PCA=3, otherwise empty)
%   E.g. if cv_max=50 and cv_increment=5, pc optimization will search over
%   PCs 1 to 50, in increments of 5 (1 5 10 15 etc)
%cv_run: numeric, determines the specific nPC that will be optimized if
%   use_PCA=3 and opt_mode=1 above (otherwise leave empty); 
%   this is determined by PC_range which is set as 1:cv_increment:maxPCs;
%   this allows for better parallelization: subjects AND specific nPCs can
%   be submitted in parallel
%sub_id: string, subject id (used for writing outputs of optimization iteratively)
%out_dir: string, path to directory where various outputs of nPC optimization will be iteratively saved (for use_PCA=3); 
%   1. To prevent out of memory errors, the outputs from each PCA is saved in its own subject/trial .mat file 
%        in a subdirectory named 'FullPCVars'; the following is saved:
%           X_mat (predictor timeseries for this trial; tp x predictors x targets), y_mat (target
%           timeseries for this trial; tp x targets), coef_mat (PCA
%           coefficients, predictors x cv_max x targets), scores_mat (PCA
%           scores, same dims), ex_mat (variance explained by each PC,
%           maxPC x targets)
%   2. The optimization result for each nPC value (determined by
%       cv_max, cv_increment and cv_run above) is also saved in out_dir, to allow for
%       easier resumption of aborted jobs (part of parallelizing nPCs as
%       well as subjects); struct 'MVAR_PCopt' is saved containing 4d nPC
%       (1 x 3 cols (opt metrics) x target regions x test trial loops)
%   *note 'out_dir' is distinct from 'outfile' which is the .mat file containing the final FC matrix



%OUTPUTS:
%Final FC information for the subject is in a single 'output' struct (mirroring what is written in 'outfile'), which contains:
%sub_restFC_avg: numeric, num_regions x num_regions; Pearson correlation (static FC) matrix,
%   estimated separately for each trial, then averaged across them
%sub_restMVAR_avg=numeric, num_predictors (regions x lags, + autoregs) x num_targets, 
%   MVAR FC matrix with target region predictor rows removed from each column
%*sub_restMVAR_viz_avg=numeric, num_predictors (regions x lags, + autoregs) x num_targets,
%   MVAR FC matrix with target region rows padded with zeros
%   Note: this variable is easier to visualize than sub_restMVAR_avg and is also input in the dynamic actflow
%   scripts
%   e.g. order of rows for a given target=region 1: t0 (if include_contemp=1), t0-1 -> up to t0-model order, 
%   region 2:t0, t0-1 -> model order, region3 etc; autoregs are appended in the last rows
%sub_varEx_avg=numeric, 1 x num_regions; the variance explained for
%   each PCA on each target region in the final estimation (after blocked
%   CV OR selection of a priori nPCs), estimated separately for each trial and then averaged across trials 
%   (used to demonstrate that final nPCs explain enough variance)
%MVAR_PCopt=struct collating results of blocked CV (use_PCA=3), contains: 
%   nPC: 4d array with dimensions: PCnum (rows); MSE, PCA variance
%       explained (opt metrics); number of target regions; number of trial folds
%   optPC: row vector with cols (1 x 3) reflecting optimization info for optimal nPC (that yielding the minimum mean MSE)
%       i.e. PCnum, MSE, PC variance explained


%% Set up parms
num_trials=size(data_input,2);
num_regions=size(data_input{1},1);
num_target_regions=length(target_regions);
num_tp=size(data_input{1},2);

%set opt variables if not specified (eg for a priori pc computation)
if ~exist('cv_max','var')
    cv_max=[];
end
if ~exist('cv_increment','var')
    cv_increment=[];
end
if ~exist('cv_run','var')
    cv_run=[];
end
if ~exist('sub_id','var')
    sub_id=[];
end
if ~exist('out_dir','var')
    out_dir=[];
end

%set MVAR parms
if include_contemp==1
    increments=model_order+1; %lags will include t0 term
else
    increments=model_order;
end

%set num_predictors
num_predictors=(num_regions-1)*increments;
if include_autoreg==1
    num_predictors=num_predictors+(model_order); %add lags for target autoregs, obviously thismust exclude t0 contemporaneous term to prevent circularity
end

%set up output variables
trial_data=zeros(num_tp,num_regions,num_trials);
sub_restFC=zeros(num_regions,num_regions,num_trials);
sub_restMVAR=zeros(num_predictors,num_target_regions,num_trials);
sub_restMVAR_viz=zeros(num_predictors+increments,num_target_regions,num_trials);

%set up 3d trial_data array (zscore each region per trial if requested); also compute corrFC (static)
tic;
for t=1:num_trials
    %rearrange into cols=tseries, rows=regions
    trial_tseries=data_input{t}';

    %zscore
    if zscore_tseries==1
        trial_tseries=zscore(trial_tseries,0,1);
    end
    
    trial_data(:,:,t)=trial_tseries;

    %compute restFC and add to sub
    r=corr(trial_tseries);
    sub_restFC(:,:,t)=r;
    
end


%% run PCA optimization (if requested)

%initialize optimization variables - *set maxPC first
nObs=num_tp-model_order; %take into account first timepoint which is model_order + 1
nVars=num_predictors; 
if nObs > nVars
    maxPC=nVars;
else
    maxPC=nObs-1; %works for cases when nObs =< nVars
end

%*set up PC range and output vars
if use_PCA==3
     PC_range=0:cv_increment:cv_max;
     %replace 0 with 1
     PC_range(1)=1;
    
    %*v4: sets which nPC is estimated for this job
    nextPC_ind=cv_run; %* sets which pc will be looped over
    
    %note num_cv_loops is num_trials-1
    num_cv_loops=num_trials-1;
    MVAR_PCopt.nPC=zeros(1,3,num_target_regions,num_cv_loops); %array with dimensions: PCnum(1:maxPC), MSE, PCA variance explained; number target regions, number trials)
    MVAR_PCopt.optPC=[];%PCnum, MSE, PC variance explained, for the optimal PCnum i.e. that which yields the minimum MSE after averaging nPC across regions and trials
        
    %initialize next* variables
    nextTrial_ind=1;
    nextTarget_ind=1;
    

else
    %initialize even if not running optimization to save output more consistently
    MVAR_PCopt.nPC=zeros(maxPC,3,num_target_regions,num_trials); %array with dimensions: PCnum(1:maxPC), MSE, PCA variance explained; number target regions, number trials)
    MVAR_PCopt.optPC=[];%PCnum, MSE, PC variance explained, for the optimal PCnum i.e. that which yields the minimum MSE after averaging nPC across regions and trials
    
    
end

%% blocked crossval
if use_PCA==3
    if opt_mode==1 %run opt

        first_tpoint=model_order+1; %must have enough lags prior to first_tpoint
        num_tp_reg=length(first_tpoint:num_tp);
        num_sources=num_regions-1;

        %**loop through and assemble pca and x/y mats, if these haven't been
        %assembled already
        %**need to save each trial PCVar to its own .mat file to avoid out of
        %memory errors
        pc_dir=[out_dir,'/FullPCvars/'];
        if ~exist(pc_dir);mkdir(pc_dir);end;

        for t=nextTrial_ind:num_trials
            pc_out=[pc_dir,sub_id,'_fullPCvars_trial',num2str(t),'.mat'];
            if ~exist(pc_out)
                %run through all trials/targets, compute PCA and  and store (more time
                %efficient than computing PCA for each nPC loop)
                coef_mat=zeros(num_predictors,cv_max,num_target_regions);%,num_trials);
                scores_mat=zeros(num_tp_reg,cv_max,num_target_regions);%,num_trials);
                ex_mat=zeros(maxPC,num_target_regions);%,num_trials);

                %also presample X and y matrices for regression (for more efficiency)
                %
                X_mat=zeros(num_tp_reg,num_predictors,num_target_regions);%,num_trials);
                y_mat=zeros(num_tp_reg,num_target_regions);%,num_trials);

                trial_tseries=trial_data(:,:,t);
                for r1=nextTarget_ind:num_target_regions
                    source_inds=1:num_regions; %for looping later
                    target_r=target_regions(r1);
                    source_inds(target_r)=[];

                    %init
                    y=[];
                    X=[];
                    for tp=first_tpoint:num_tp
                        %assign y
                        y=[y;trial_tseries(tp,target_r)];

                        %set up X, format is:
                        %1. all other regions over all lags 1->n (incl contemp if
                        %applicable; e.g. region1: t0, t0-m -> model order, region 2:
                        %t0, t0-m -> model order etc
                        %2. autoregs over lags (if applicable)
                        %3. constant
                        tpoint_X=[];
                        for rr=1:length(source_inds)
                            rr_ind=source_inds(rr);

                            %add contemp if necessary
                            if include_contemp==1
                                tpoint_X=[tpoint_X,trial_tseries(tp,rr_ind)];
                            end

                            %add lags
                            for lag=1:model_order
                                tpoint_X=[tpoint_X,trial_tseries(tp-lag,rr_ind)];                        
                            end
                        end
                        %add autoregs for target over lags
                        if include_autoreg==1
                            for lag=1:model_order
                                tpoint_X=[tpoint_X,trial_tseries(tp-lag,target_r)];                        
                            end
                        end

                        %assign to X
                        X=[X;tpoint_X];               

                    end

                    %demean prior to PCA
                    X=X-repmat(mean(X,1),size(X,1),1);
                    %run pca (up till max range)
                    [Loadings,Scores,~,~,explained] = pca(X,'NumComponents',cv_max);

                    %add to mats
    %                     coef_mat(:,:,r1,t)=Loadings;
    %                     scores_mat(:,:,r1,t)=Scores;
    %                     ex_mat(:,r1,t)=explained;
    %                     X_mat(:,:,r1,t)=X;
    %                     y_mat(:,r1,t)=y;

                    coef_mat(:,:,r1)=Loadings;
                    scores_mat(:,:,r1)=Scores;
                    ex_mat(:,r1)=explained;
                    X_mat(:,:,r1)=X;
                    y_mat(:,r1)=y;

                    disp(['Time taken to assemble trial x target vars for PCA opt, for target',num2str(r1),' trial',num2str(t),' = ',num2str(toc)]);           
                end
                %save
                save(pc_out,'coef_mat','scores_mat','ex_mat','X_mat','y_mat','-v7.3');
    %             else
    %                 load(pc_out);
            end
        end

        %loop through number of PCs and number of targets and compute
        %opt metrics: cross-trial MSE and PCexplained
        for pc=nextPC_ind
            nPCs=pc;
            
            %set output file for this nPC
            nPC_out=[out_dir,'/',sub_id,'_out_nPC',num2str(nPCs),'.mat'];

            for r1=nextTarget_ind:num_target_regions

                %**loop through folds training/testing MVAR model
                for t=1:num_cv_loops
                    %assign train/test trial
                    train_trial=t;
                    test_trial=t+1;

                    %load in train trial and assign appropriate vars
                    train_mat=[pc_dir,sub_id,'_fullPCvars_trial',num2str(train_trial),'.mat'];
                    load(train_mat);
                    PCAScores=scores_mat(:,1:nPCs,r1);
                    PCAScores=[PCAScores,ones(size(PCAScores,1),1)];
                    PCALoadings=coef_mat(:,1:nPCs,r1);
                    PCAEx=sum(ex_mat(1:nPCs,r1)); %cumulative explained PCs
                    %X=X_mat(:,:,r1,train_ind);
                    y=y_mat(:,r1);

                    %set up test vars
                    test_mat=[pc_dir,sub_id,'_fullPCvars_trial',num2str(test_trial),'.mat'];
                    load(test_mat);
                    X_test=X_mat(:,:,r1);
                    y_test=y_mat(:,r1);

                    %regress
                    b=regress(y,PCAScores); %stats(1)=Rsquared
                    %b(length(b))=[]; %remove constant
                    b_constant=b(end); %keep constant for later yhat prediction
                    b(end)=[];
                    %transform back to predictors
                    b=PCALoadings*b;  

                    %loop through test tp and use b to predict test data, and store actual and pred data
                    actual_data=[];
                    pred_data=[];
                    for tt=1:num_tp_reg
                        actual_data=[actual_data;y_test(tt)];                     
                        %predict
                        pred=dot(X_test(tt,:),b)+b_constant;
                        pred_data=[pred_data;pred];
                    end
                    %compute mse
                    err = immse(actual_data,pred_data);
                    pc_dat=[nPCs,err,PCAEx];
                    MVAR_PCopt.nPC(1,:,r1,t)=pc_dat;

                    %update counter
                    %nextPC_ind=nextPC_ind+1;

                    %save outfile - *no point saving empty sub_* vars - might
                    %increase computation time
                    %save(outfile,'MVAR_PCopt','nextTrial_ind','nextTarget_ind','nextPC_ind','-v7.3');

                    disp(['Time taken to estimate OPTIMIZED MVAR for region',num2str(r1),' trial loop',num2str(t),' pc',num2str(nPCs),' = ',num2str(toc)]);           

                end
            end
            %*save output for this pc
            save(nPC_out,'MVAR_PCopt','-v7.3');
        end
        
    elseif opt_mode==2
        %load in nPC_out over full PC_range
        MVAR_PCopt_full=zeros(length(PC_range),3,num_target_regions,num_cv_loops); %array with dimensions: PCnum(1:maxPC), MSE, PCA variance explained; number target regions, number trials)
        for n=1:length(PC_range)
            nPCs=PC_range(n);
            nPC_out=[out_dir,'/',sub_id,'_out_nPC',num2str(nPCs),'.mat'];
            load(nPC_out);
            MVAR_PCopt_full(n,:,:,:)=MVAR_PCopt.nPC;
        end

        %*reassign to MVAR_PCopt.nPC for consistency with below
        MVAR_PCopt.nPC=MVAR_PCopt_full;

        %*identify optimal nPCs (min over avg MSE across trials and targets) and add that row to MVAR_PCopt.optPC
        mean_opts=mean(mean(MVAR_PCopt.nPC,4),3);
        [opt_mse,ind]=min(mean_opts(:,2));
        MVAR_PCopt.optPC=mean_opts(ind,:);
    end
    
end

%% loop through target regions and compute restMVAR via PCA regression
%can run all versions of use_PCA here i.e. for 2 and 3 optimization, this
%will run the optimal nPCs

%note - using 'regress' function rather than regstats (which adds a col of ones by default),
%as regress seems to have more stringent control for rank deficiency


%**set nPCs based on use_PCA
if opt_mode==2
    if use_PCA==3
        nPCs=MVAR_PCopt.optPC(1); %first column = optimal number of PCs after crossval
    elseif use_PCA > 3
        nPCs=use_PCA;
    end
    
    %also save PCA var explained
    sub_varEx=zeros(num_trials,num_target_regions);

    %*loop through trials and target regions and compute pca regression -
    for t=1:num_trials
        trial_MVAR=[];
        trial_MVAR_viz=[];
        trial_tseries=trial_data(:,:,t);
        %
        trial_varEx=[];
        for r1=1:num_target_regions
            first_tpoint=model_order+1; %must have enough lags prior to first_tpoint

            source_inds=1:num_regions; %for looping later
            target_r=target_regions(r1);
            source_inds(target_r)=[];

            %init
            y=[];
            X=[];
            for tp=first_tpoint:num_tp
                %assign y
                y=[y;trial_tseries(tp,target_r)];

                %set up X, format is:
                %1. all other regions over all lags 1->n (incl contemp if
                %applicable; e.g. region1: t0, t0-m -> model order, region 2:
                %t0, t0-m -> model order etc
                %2. autoregs over lags (if applicable)

                tpoint_X=[];
                for rr=1:length(source_inds)
                    rr_ind=source_inds(rr);

                    %add contemp if necessary
                    if include_contemp==1
                        tpoint_X=[tpoint_X,trial_tseries(tp,rr_ind)];
                    end

                    %add lags
                    for lag=1:model_order
                        tpoint_X=[tpoint_X,trial_tseries(tp-lag,rr_ind)];                        
                    end
                end
                %add autoregs for target over lags
                if include_autoreg==1
                    for lag=1:model_order
                        tpoint_X=[tpoint_X,trial_tseries(tp-lag,target_r)];                        
                    end
                end

                %assign to X
                X=[X;tpoint_X];               

            end

            %demean prior to pca
            X=X-repmat(mean(X,1),size(X,1),1);
            %run pca
            [PCALoadings,PCAScores,~,~,explained] = pca(X,'NumComponents',nPCs);

            %check for rank deficiency of PCAScores (rank deficiency of
            %sub_timeseries translates to rank deficiency of PCAScores)
            PCA_rank=rank(PCAScores);
            if PCA_rank<nPCs
                PCAScores=PCAScores(:,1:PCA_rank);
                PCALoadings=PCALoadings(:,1:PCA_rank);
                %Provide warning to user
                disp(['****WARNING: rank of PCAScores (rank=',num2str(PCA_rank),') is lower than retained PCs (',num2str(size(PCAScores,2)),')']);
                disp('*****Setting retained PCs to equal PCA rank for this trial/region model, but consider choosing a priori lower retained PCs (using use_PCA) for better consistency across trials/regions');
            end
            %regress
            PCAScores=[PCAScores,ones(size(PCAScores,1),1)]; %add constant
            b=regress(y,PCAScores); 
            b(end)=[]; %remove constant
            %transform back to predictors
            b=PCALoadings*b;

            %for mvar_viz: re-arrange to consistent format over regions, by assigning 0 over lags to target (preserve autoregs at end of each col)
            %permits visualization       
            row_start=increments*(target_r-1)+1;
            row_end=target_r*increments; %no need to include contemp, just need to insert a constant that differentiates
            %insert into betas
            b_viz=b;
            if target_r==1
                b_viz=[zeros(increments,1);b_viz];
            else
                b_viz=[b_viz(1:row_start-1);zeros(increments,1);b_viz(row_start:end)];
            end

            %add to trial_MVAR
            trial_MVAR=[trial_MVAR,b];
            trial_MVAR_viz=[trial_MVAR_viz,b_viz];
            trial_varEx=[trial_varEx,sum(explained(1:nPCs))];

            disp(['Time taken to estimate MVAR for region',num2str(r1),' trial',num2str(t),' = ',num2str(toc)]);           

        end
        %add to sub_restMVAR
        sub_restMVAR(:,:,t)=trial_MVAR;
        sub_restMVAR_viz(:,:,t)=trial_MVAR_viz;
        sub_varEx(t,:)=trial_varEx;


    end
    %average across trials and add to output
    output.sub_restFC_avg=mean(sub_restFC,3);
    output.sub_restMVAR_avg=mean(sub_restMVAR,3);
    output.sub_restMVAR_viz_avg=mean(sub_restMVAR_viz,3);
    output.sub_varEx_avg=mean(sub_varEx,1);
    output.MVAR_PCopt=MVAR_PCopt;

    %% write output
    sub_restFC_avg=output.sub_restFC_avg;
    sub_restMVAR_avg=output.sub_restMVAR_avg;
    sub_restMVAR_viz_avg=output.sub_restMVAR_viz_avg;
    sub_varEx_avg=output.sub_varEx_avg;
    MVAR_PCopt=output.MVAR_PCopt; %contains nPC array, and optPC (PCnum, MSE, PC variance explained)

    %save each subject separately
    if use_PCA==3
        save(outfile,'sub_restFC_avg','sub_restMVAR_avg','sub_restMVAR_viz_avg','sub_varEx_avg','MVAR_PCopt','-v7.3');
    else
        save(outfile,'sub_restFC_avg','sub_restMVAR_avg','sub_restMVAR_viz_avg','sub_varEx_avg','-v7.3');
    end
    
end

end

