clear all; close all; clc;


%% Required inputs

sensor = '...'; % Sensor name
ratio = ...; % The resolution scale which elapses between MS and PAN

input_image_path = '...'; % The path of the .mat file which contains the MS and PAN images
pansharpened_image_path = '...'; % The path of pansharpened image

%% Auxiliary inputs

Qblocks_size = 32; % The windows size on which calculate the Q2n index
flag_cut_bounds = 0; % Cutting flag for obtaining "valid" image to which apply the metrics
dim_cut = 21; % Cutting dimension for obtaining "valid" image to which apply the metrics

%% Data opening and conversion 

original_data = load(input_image_path);
pansharpend_data = load(pansharpened_image_path);

I_MS_LR = double(original_data.I_MS_LR);
I_PAN = double(original_data.I_PAN);
I_GT = double(original_data.I_MS_LR);
P = double(pansharpend_data.I_MS);


%% Auxiliary computation
I_MS = interp23tap(I_MS_LR,ratio);


%% Metrics computation
P_repro = resize_w_mtf(P,I_MS_LR,I_PAN,sensor,ratio);
[R_Q2n, R_Q, R_SAM, R_ERGAS] = consistency_metrics_evaluation(P_repro,I_MS_LR,ratio,Qblocks_size,flag_cut_bounds,dim_cut)
D_RHO = D_rho(P,I_PAN,double(ratio/2))


