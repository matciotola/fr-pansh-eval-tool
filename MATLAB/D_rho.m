function D_RHO = D_rho(I,J,half_width)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Spatial Quality Index based on local cross-correlation.
% 
% Interface:
%           D_RHO = D_rho(I,J,half_width)
% 
% Inputs:
%           I:              First image;
%           J:              Second image;
%           half_widht:     The semi-size of the window on which calculate the cross-correlation; 
% 
% Outputs:
%           D_RHO:          The D_rho index
% References:
%           [Scarpa21]      Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
%                           arXiv preprint arXiv:2108.06144
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rho = local_cross_correlation(I, J, half_width);
rho(rho>1.0)=1.0;
rho(rho<-1.0)=-1.0;

rho = 1.0 - rho;
D_RHO = mean2(rho);

end