function rho = local_cross_correlation(I,J,half_width)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Cross-Correlation Field computation.
% 
% Interface:
%           rho = local_cross_correlation(I,J,half_width)
% 
% Inputs:
%           I:              First image;
%           J:              Second image;
%           half_widht:     The semi-size of the window on which calculate the cross-correlation; 
% 
% Outputs:
%           rho:            The cross-correlation map between I and J images
% References:
%           [Scarpa21]      Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
%                           arXiv preprint arXiv:2108.06144
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    w = ceil(half_width);
    ep = single(1e-20);

    I = single(padarray(I,[w,w]));      J = single(padarray(J,[w,w]));
    I_cum = integralImage(I);   J_cum = integralImage(J);
    I_cum = I_cum(2:end,2:end,:); J_cum = J_cum(2:end,2:end,:);

    I_mu = (I_cum((2*w+1):end,(2*w+1):end,:) - I_cum(1:(end-2*w),(2*w+1):end,:) - ...
           I_cum((2*w+1):end,1:(end-2*w),:) + I_cum(1:(end-2*w),1:(end-2*w),:))./ ...
           (4*w^2);


    J_mu = (J_cum((2*w+1):end,(2*w+1):end,:) - J_cum(1:(end-2*w),(2*w+1):end,:) -...
           J_cum((2*w+1):end,1:(end-2*w),:) + J_cum(1:(end-2*w),1:(end-2*w),:))./...
           (4*w^2);

    I = I(w+1:end-w,w+1:end-w,:)-I_mu; 
    J = J(w+1:end-w,w+1:end-w,:)-J_mu;

    I = padarray(I,[w, w]);    J = padarray(J,[w, w]);

    I2_cum = integralImage(I.^2); I2_cum = I2_cum(2:end,2:end,:);  
    J2_cum = integralImage(J.^2); J2_cum = J2_cum(2:end,2:end,:);  
    IJ_cum = integralImage(I.*J); IJ_cum = IJ_cum(2:end,2:end,:); 

    sig2_IJ_tot = IJ_cum((2*w+1):end,(2*w+1):end,:) - IJ_cum(1:(end-2*w),(2*w+1):end,:) - ...
           IJ_cum((2*w+1):end,1:(end-2*w),:) + IJ_cum(1:(end-2*w),1:(end-2*w),:);

    sig2_I_tot = I2_cum((2*w+1):end,(2*w+1):end,:) - I2_cum(1:(end-2*w),(2*w+1):end,:) - ...
           I2_cum((2*w+1):end,1:(end-2*w),:) + I2_cum(1:(end-2*w),1:(end-2*w),:);

   sig2_J_tot = J2_cum((2*w+1):end,(2*w+1):end,:) - J2_cum(1:(end-2*w),(2*w+1):end,:) - ...
           J2_cum((2*w+1):end,1:(end-2*w),:) + J2_cum(1:(end-2*w),1:(end-2*w),:);
    
    sig2_I_tot = max(ep,sig2_I_tot); sig2_J_tot = max(ep,sig2_J_tot); %%%% PATCH

    rho = sig2_IJ_tot./((sig2_I_tot.*sig2_J_tot).^0.5+ep);
    %L = 1 - mean2(L);
    
end
