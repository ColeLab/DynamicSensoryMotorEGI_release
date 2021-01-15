function [pred,actual]=dynamic_actflow(activity,restMVAR,target_inds,MVAR_increments,cond_inds,include_contemp,include_autoreg,exclude_target_net,zscore_sub,regress_targetAct,titrate_lags,pairwise_lags,actflow_source_contemp,actflow_target_autoregs)
%Author: Ravi Mill, rdm146@rutgers.edu
%Last update: Jan 15th 2021

%DESCRIPTION: function that generates timepoint-by-timepoint taask activation predictions via dynamic
%activity flow modeling (using source EEG task and MVAR FC inputs).
%For Dynamic Activity flow project: Mill et al, "Causal emergence of task information from
%dynamic network interactions in the human brain"

%INPUTS:
%activity: numeric 3d array; task activation timeseries (trials,all regions,timepoints)
%restMVAR: numeric 2d array; MVAR connectivity weight matrix (predictors, target i.e. to-be-predicted rgions); 
    %predictors=source regions*MVAR_increments, and if include_autoreg=1, last set of rows 
    %contains autoreg (lagged, self-coupling) terms for each target region;
    %note also that for each target region's column, row indices corresponding to that target region 
    %are padded with zeros (which are removed before running actflow below)
    %*Note also that the order of regions should match for columns of activity (i.e. source 1, 2...n sources) and rows (predictors) of
    %restMVAR (e.g. source 1 lag t0-1, source 1 lag t0-2...lag t0-n; source
    %2 lag t0-1...etc with target autoregs appended last)
%target_inds: numeric column vector, contains indices for targets of actflow
    %prediction; enables identification of targets and sources in restMVAR and activity variables
%MVAR_increments: numeric, used to index restMVAR for each region based on
    %model_order of MVAR and whether contemporaneous (t0) terms are
    %included in that matrix (i.e. if include_contemp=1, then MVAR_ncrements=model_order+1)
%cond_inds: numeric column vector, coding for conditions in activity matrix (with length corresponding to number 
    %of trials i.e. row dimension); 1=cond1 trial, 2=cond2 trial etc; 
    %*note this is currently not used in the function (each trial is predicted separately without reference to condition)
%include_contemp: binary; 1=source contemporaneous (t0) terms are included in restMVAR
%include_autoreg: binary; 1=autoreg (lagged/self-coupling) terms for each target region are included
    %in restMVAR (in the last rows)
%exclude_target_net: binary; 1=exclude regions from the same network as the 
    %target from the actflow predictions (i.e. all regions specified in
    %target_inds)
%zscore_sub: binary; 1=zscore predicted and actual timeseries for each region
    %and each trial prior to storing in pred/actual output variables (NOT RECOMMENDED)
%regress_targetAct: binary; 1=regress out the target task activation
    %timeseries for a given trial from all source activation timeseries prior to computing actflow; 
    %*RECOMMENDED to control for field spread contaminants in
    %source EEG/MEG (i.e. instantaneous target t0 -> source t0 effects)
%titrate_lags: numeric; if set to a nonzero value, this will remove the specified lag from
    %the FC and activity terms for each dot product operation that computes actflow. 
    %e.g. 1=remove t0-1 lag terms from atflow. Allows assessment of how far back in
    %the past we can still accurately predict the future.
%pairwise_lags: numeric, if set to a nonzero value, this will confine the
    %actflow prediction to specified lags only e.g. 1=compute dot product using
    %t0-1 lag terms only. The idea again (as an alternative to titrate_lags) is
    %to see how far back in the past we can still accurately predict the future.
%actflow_source_contemp: binary; 1=preserve source contemp (t0) terms in
    %the actflow dot product (assuming these are present in the restMVAR);
    %RECOMMENDED=0 to avoid field spread
%actflow_target_autoregs: binary; 1= preserve target autoreg terms (t0-1)
    %in the actflow dot product (assuming these are present in the
    %restMVAR); RECOMMENDED=1 to account for self-coupling influences

%OUTPUTS:
%pred: 3d array containing actflow-predicted task timeseries (trials, target regions,tpoints)
%actual: 3d array containing actual task timeseries (trials, target regions,tpoints)

    
numTasks=length(unique(cond_inds));
numTrials=size(activity,1);
numRegions_all=size(activity,2);
numRegions_targets=length(target_inds);


%set first timepoint to estimate actflow (based on model order)
if include_contemp==1 && actflow_source_contemp==0
    model_order=MVAR_increments-1; %excludes the first lag term for each region i.e. t0, because this will be removed from restMVAR
else
    model_order=MVAR_increments;
end
first_tpoint=model_order+1;
numTpoints=length(first_tpoint:size(activity,3));

%remove contemporaneous elements from restMVAR - excludes last terms that correspond to
%target autoregs
%*check if include_source_contemp==1, in which case these terms are
%included
if include_contemp==1 && actflow_source_contemp==0
    contemp_inds=1:MVAR_increments:numRegions_all*MVAR_increments;
    restMVAR(contemp_inds,:)=[];
end

%remove autoregs (if necessary)
if include_autoreg==1 && actflow_target_autoregs==0
    if include_contemp==1
        restMVAR(end-(MVAR_increments-1)+1:end,:)=[]; %last ind to retain will vary if restMVAR includes contemp terms for sources
    elseif include_contemp==0
        restMVAR(end-MVAR_increments+1:end,:)=[];
    end

end

%create vector coding for regions in the restMVAR matrix
MVAR_regions=repmat(1:numRegions_all,model_order,1);
MVAR_regions=MVAR_regions(:);

%Setup for prediction
pred=zeros(numTrials,numRegions_targets,numTpoints);
actual=zeros(numTrials,numRegions_targets,numTpoints);
regionNumList=1:numRegions_all;

%set pairwise_lags if it does not exist
if exist('pairwise_lags','var')==0
    pairwise_lags=0;
end

%titrate_lags and pairwise_lags cannot both be 1
if titrate_lags==1 && pairwise_lags==1
    error('titrate_lags and pairwise_lags cannot both be set to 1!');
end

for trialNum=1:numTrials

    %Get this subject's activation pattern for this trial
    %taskActVect=activity(trialNum,:,:);
    for regionNum=1:numRegions_targets
        
        %assign target and source inds
        target=target_inds(regionNum);
        sources=regionNumList;
        
        %*exclude target and also set whether to exclude same network regions (other targets) from source
        if exclude_target_net==1
            [~,ind]=intersect(sources,target_inds);
            sources(ind)=[];
            
        elseif exclude_target_net==0
            sources(target)=[];%Hold out region whose activity is being predicted
        end
                
        %set up activity
        target_act=activity(trialNum,target,:);
        source_act=activity(trialNum,sources,:);
        
        %**regress out targetAct from all sources if this parm is requested
        %meant to control for field spread
        if regress_targetAct==1
            %need to add column of ones (affects betas/resid if data is not
            %normalized)
            reg_target=[squeeze(target_act(1,1,:)),ones(size(target_act,3),1)];
            for ss=1:size(source_act,2)
                [~,~,resid]=regress(squeeze(source_act(1,ss,:)),reg_target);
                %replace source_act with residuals
                source_act(1,ss,:)=resid;
            end       
        end
        
        %set up FC
        target_FC=restMVAR(:,regionNum);
        
        %exclude target_regions from targetFC
        if exclude_target_net==1
            %exclude all target network regions
            [~,ind]=intersect(MVAR_regions,target_inds);
            
        elseif exclude_target_net==0
            %only exclude current target region (target)
            [~,ind]=intersect(MVAR_regions,target);
        end
        %intersect only find first instance
        rem_inds=[];
        for e=1:length(ind)
            e_start=ind(e);
            e_end=e_start+model_order-1;
            rem_inds=[rem_inds;(e_start:e_end)'];
        end  
        target_FC(rem_inds,:)=[];
        
        %make activity prediction at each timepoint, for the current target
        %region, based on lagged activation + FC terms        
        %set tpoint counter
        t0=first_tpoint;
        actual_region=[];
        pred_region=[];
        for tpoint=1:numTpoints
            %reassign target_FC on each loop (given potential to adjust it
            %by titrate_lags
            target_FC_tpoint=target_FC;
            %assign actual
            %actual(tpoint,regionNum,trialNum)=target_act(1,1,t0);
            actual_region=[actual_region;target_act(1,1,t0)];
            
            %compute pred - need to assemble act and FC terms
            %appropriately i.e. lining up lags
            %targetFC rows=region1(all model_orders t0-1to10),region2 (all
            %model orders etc)
            activity_lags=[];
            for m=1:model_order
                m_act=source_act(:,:,t0-m);
                activity_lags=[activity_lags;m_act];
            end
            %vectorize so it matches up with FC
            activity_lags=activity_lags(:);
            
            %*add target lags to end of activity lags too (if requested)
            if include_autoreg==1 && actflow_target_autoregs==1
                target_lags=[];
                for m=1:model_order
                    m_act=target_act(:,:,t0-m);
                    target_lags=[target_lags;m_act];
                end
                %vectorize and add to activity lags
                target_lags=target_lags(:);
                activity_lags=[activity_lags;target_lags];
                
            end
            
            %**titrate lags (if specified)
            if titrate_lags~=0
                %generate vector of specified lags
                m_orders=1:model_order;
                lags_to_exclude=find(m_orders<titrate_lags);
                num_sources=size(sources,2);
                %generate vector coding for lag inds to exclude from
                %activity_lags and target_FC
                lag_vec=[];
                for mm=1:length(lags_to_exclude)
                    mm_lag=lags_to_exclude(mm);
                    mm_vec=mm_lag:model_order:model_order*num_sources;
                    lag_vec=[lag_vec,mm_vec];
                    
                end
                %remove from FC/activity
                target_FC_tpoint(lag_vec)=[];
                activity_lags(lag_vec)=[];
                
            elseif pairwise_lags~=0
                %generate vector of specified lags
                %m_orders=1:model_order;
                %lags_to_exclude=find(m_orders<titrate_lags);
                num_sources=size(sources,2);
                %generate vector coding for lag inds to exclude from
                %activity_lags and target_FC
                lag_vec=pairwise_lags:model_order:model_order*num_sources;

                %remove from FC/activity
                target_FC_tpoint=target_FC_tpoint(lag_vec);
                activity_lags=activity_lags(lag_vec);
                
            end
            
            %compute actflow
            %store in pred
            %pred(tpoint,regionNum,trialNum)=dot(target_FC,activity_lags);
            pred_region=[pred_region;dot(target_FC_tpoint,activity_lags)];
            
            %update t0
            t0=t0+1;
            
        end
        %add to counter after zscoring to correct for scaling differences
        if zscore_sub==1
            actual(trialNum,regionNum,:)=zscore(actual_region);
            pred(trialNum,regionNum,:)=zscore(pred_region);
        else
            actual(trialNum,regionNum,:)=actual_region;
            pred(trialNum,regionNum,:)=pred_region;
        end
        
    end
end


end

