function [H, E_v, F_v, result_all]= EMR_CA(Z_v, num_dim, r, num_cluster, max_iter, Y)


num_view = size(Z_v,2); % or 1
[num_instance num_anchor] = size(Z_v{1});
for vv = 1:num_view
    Deta_v{vv} = sum(Z_v{vv},1); % 1*num_anchor
    Deta_inv2_v{vv} = diag(1./(sqrt(Deta_v{vv})+eps)); % num_anchor*num_anchor
    % Init E_v
    if num_dim == num_anchor
        E_v{vv} = Deta_inv2_v{vv}; %  num_dim == num_anchor
    else
        E_v{vv} = orth(randn(num_dim,num_anchor))*Deta_inv2_v{vv}; %  num_dim > num_anchor
    end  
    % Init F_v
    temp = E_v{vv}*Z_v{vv}';
    [Ur, ~, Vr] = svds(temp,num_cluster);
    F_v{vv} = Ur;
end

for iter_ii=1:max_iter
    %% optimize alpha
    if iter_ii==1
        % Init alpha
        alpha = ones(1,num_view)/num_view;
        sum_alpha = 1;
       %% optimize H
        options = optimset('Algorithm','interior-point-convex','Display','off'); % Algorithm é»˜è®¤ä¸? interior-point-convex
        for ii=1:num_instance % parfor
            M = 0;
            ff=0;
            for vv = 1:num_view
                al2 = alpha(vv)^r;
                M = M + 2 * al2 *  eye(num_cluster);
                ff = ff - 2 * al2 * Z_v{vv}(ii,:) * E_v{vv}'*F_v{vv};
            end
            H(ii,:) = quadprog(M,ff',[],[],ones(1,num_cluster),1,zeros(num_cluster,1),ones(num_cluster,1),[],options);
        end
    else
        sum_delte =0;
        for vv = 1:num_view
            temp1 = Z_v{vv}*E_v{vv}' - (H)*F_v{1,vv}';
            sigma2(1,vv) = sum(sum(temp1.^2))/(2*num_instance);
            alpha(1,vv) = sum(exp(-sum(temp1.^2)./(2*sigma2(1,vv))));
%             alpha(1,vv) = sum(exp(-sum(diag(p_v{vv}) * (temp1.^2))./(2*sigma2(1,vv))));
%             alpha(1,vv) = sum(sum(diag(p_v{vv}) * (temp1.^2)));
            sum_delte = sum_delte + (r*alpha(1,vv))^(1/(1-r));
        end
        sum_alpha = 0;
        for vv = 1:num_view
            alpha(1,vv) = ((r*alpha(1,vv))^(1/(1-r)))/sum_delte;
            sum_alpha = sum_alpha + alpha(1,vv);
        end
    end
    
   %% optimize p
    obj = 0;
    for vv = 1:num_view
        temp1 = Z_v{vv}*E_v{vv}' - (H)*F_v{1,vv}';
        temp1 = temp1.^2;
        sigma2(1,vv) = sum(sum(temp1))/(2*num_instance);
        p_v{vv} = exp(-sum(temp1,2)./(sigma2(1,vv)));
        % mean_pv{vv} = mean(p_v{vv})./(sigma2(1,vv));
        p_v{vv} = p_v{vv}./(sigma2(1,vv));
        
        temp1 = Z_v{vv}*E_v{vv}' - (H)*F_v{1,vv}';
        temp1 = sum(temp1.^2,2);
        temp1 = exp(-temp1./(2*sigma2(1,vv)^2));
        obj = obj + alpha(1,vv)^r*sum(temp1);
    end
    obj_all(iter_ii) = obj;
    %% optimize H
    options = optimset( 'Algorithm','interior-point-convex','Display','off'); % Algorithm é»˜è®¤ä¸? interior-point-convex
    for ii=1:num_instance % parfor
        M = 0;
        ff = 0;
        for vv = 1:num_view
            al2 = alpha(vv)^r;
            M = M + 2 * al2 * p_v{vv}(ii) * eye(num_cluster);
            ff = ff - 2 * al2 * p_v{vv}(ii) * Z_v{vv}(ii,:) * E_v{vv}'*F_v{vv};
        end
        H(ii,:) = quadprog(M,ff',[],[],ones(1,num_cluster),1,zeros(num_cluster,1),ones(num_cluster,1),[],options);
    end
    %% optimize F_v
    for vv = 1:num_view % parfor
        temp = E_v{vv}*Z_v{vv}'*diag(p_v{vv})*(H);
        [Ur, ~, Vr] = svds(temp,num_cluster);
        F_v{vv} = Ur*Vr';
    end 
    %% optimize E_v
    parfor vv = 1:num_view
        temp = F_v{vv}*(H)'*Z_v{vv};
        [Ur, ~, Vr] = svds(temp,num_anchor);
        E_v{vv} = Ur*Vr'*Deta_inv2_v{vv};
    end 
    %%
    [non ,Y_Hpred] = max(H,[],2);
    % [ACC MIhat Purity]
    temp_res = Clustering8Measure(Y, Y_Hpred);
    result_all(iter_ii,:) = temp_res;

   
end

