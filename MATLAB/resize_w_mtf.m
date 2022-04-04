function F_LR = resize_w_mtf(F,MS,PAN,sensor,ratio)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           resizing of the fused product F using MTF and PAN alignment
% 
% Interface:
%           F_LR = resize_w_mtf(F,MS,PAN,sensor,ratio)
% 
% Inputs:
%           F:          Pansharpened image;
%           MS:         Original Multi-Spectral image;
%           PAN:        Original PAN imge;
%           sensor:     String for type of sensor (e.g. 'WV2','IKONOS');
%           ratio:      Scale ratio between MS and PAN. Pre-condition: Integer value.
% 
% Outputs:
%           Q_avg:      Q index averaged on all bands.
% 
% References:
%           [Wang02]    Z. Wang and A. C. Bovik, A universal image quality index, IEEE Signal Processing Letters, vol. 9, no. 3, pp. 8184, March 2002.
%           [Vivone20]  G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                       IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
%           [Scarpa21]  Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
%                       arXiv preprint arXiv:2108.06144
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Nb = size(F,3);
    MTF_vars = generate_MTF_variables(ratio,sensor,Nb,PAN,MS);

    h = MTF_vars.MTF_kern;
    pad = (size(h,1)-1)/2;
    
    for b = 1:Nb
        F(:,:,b) = imfilter(F(:,:,b),h(:,:,b),'replicate');
    end

    F_LR = zeros(size(MS));

    for b = 1:Nb
        r = MTF_vars.r(b); c = MTF_vars.c(b);
        % r = 3; c = 3;
        F_LR(:,:,b) = F(r:4:end,c:4:end,b);
    end

end