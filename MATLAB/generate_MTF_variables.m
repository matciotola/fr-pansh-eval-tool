function MTF_vars = generate_MTF_variables(ratio,sensor,nbands,PAN,MS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Compute the estimated MTF filter kernels for the supported 
%           satellites and calculate the spatial bias between each 
%           Multi-Spectral band and the Panchromatic (to implement the 
%           coregistration feature).
% 
% Interface:
%           MTF_vars = generate_MTF_variables(ratio,sensor,nbands,PAN,MS)
% 
% Inputs:
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value;
%           sensor:         The name of the satellites which has provided the images;
%           nbands:         Number of Spectral Bands of MS image; 
%           PAN:            Panchromatic image;
%           MS:             Multi-Spectral image;
% 
% Outputs:
%           MTF_vars:       Dictionary composed of the estimated MTF filter 
%                           kernel, row and column indexes vectors for 
%                           co-registration task.
% References:
%           [Scarpa21]      Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
%                           arXiv preprint arXiv:2108.06144
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    h = genMTF(ratio, sensor, nbands);
    
    P = imfilter(PAN,h(:,:,1),'replicate');
    
    rho = zeros (ratio,ratio,nbands); 
    for i=1:ratio
        for j = 1:ratio
            for b = 1:nbands
                rho(i,j,b) = mean2(local_cross_correlation(MS(:,:,b),P(i:ratio:end,j:ratio:end),2));
            end
        end
    end
    
    for b = 1:nbands
        x = rho(:,:,b);
        [mas,pos] = max(x(:));
        [r(b), c(b)] = ind2sub([ratio ratio],pos);
    end

    MTF_vars.MTF_kern = h;
    MTF_vars.r = r;
    MTF_vars.c = c;
    MTF_vars.rho = rho;
    
end