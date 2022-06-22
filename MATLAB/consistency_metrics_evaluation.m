%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Reduced resolution quality indexes. 
% 
% Interface:
%           [Q2n_index, Q_index, SAM_index, ERGAS_index] = indexes_evaluation(I_F,I_GT,ratio,L,Q_blocks_size,flag_cut_bounds,dim_cut,th_values)
%
% Inputs:
%           I_F:                Fused Image;
%           I_GT:               Ground-Truth image;
%           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value;
%           Q_blocks_size:      Block size of the Q-index locally applied;
%           flag_cut_bounds:    Cut the boundaries of the viewed Panchromatic image;
%           dim_cut:            Define the dimension of the boundary cut;
%
% Outputs:
%           Q2n_index:          Q2n index.
%           Q_index:            Q index;
%           SAM_index:          Spectral Angle Mapper (SAM) index;
%           ERGAS_index:        Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS) index;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Q2n_index, Q_index, SAM_index, ERGAS_index] = indexes_evaluation(I_F,I_GT,ratio,Q_blocks_size,flag_cut_bounds,dim_cut)

if flag_cut_bounds
    I_GT = I_GT(dim_cut:end-dim_cut,dim_cut:end-dim_cut,:);
    I_F = I_F(dim_cut:end-dim_cut,dim_cut:end-dim_cut,:);
end


Q2n_index = q2n(I_GT,I_F,Q_blocks_size,Q_blocks_size);
Q_index = Q(I_GT,I_F);
SAM_index = SAM(I_GT,I_F);
ERGAS_index = ERGAS(I_GT,I_F,ratio);

end
