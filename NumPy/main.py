import argparse
from metrics import ReproMetrics, DRho

def main_metrics(args):
    from scipy import io
    import torch
    import os

    test_path = args.input
    fused_path = args.fused
    sensor = args.sensor
    ratio = args.ratio
    out_dir = args.out_dir
    gpu_number = str(args.gpu_number)
    use_cpu = args.use_cpu
    save_outputs_flag = args.save_outputs
    show_results_flag = args.show_results

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")

    to_fuse = io.loadmat(test_path)
    outputs = io.loadmat(fused_path)['I_MS'].astype('float32')

    I_PAN = to_fuse['I_PAN'].astype('float32')
    I_MS = to_fuse['I_MS_LR'].astype('float32')

    nbands = I_MS.shape[-1]

    r_q2n, r_q, r_sam, r_ergas = ReproMetrics(outputs, I_MS, I_PAN, sensor, ratio, 32, 32)
    D_rho = DRho(outputs, I_PAN, ratio)

    if save_outputs_flag:
        io.savemat(
            out_dir + fused_path.split(os.sep)[-1].split('.')[0] + '_Coregistered_Reprojected_Metrics.mat',
            {
                'ReproQ2n': r_q2n,
                'ReproERGAS': r_ergas,
                'ReproSAM': r_sam,
                'ReproQ': r_q,
                'D_rho': D_rho,
            }
        )

        print("ReproQ2n:   %.5f \n"
              "ReproERGAS: %.5f \n"
              "ReproSAM    %.5f \n"
              "ReproQ:     %.5f \n"
              "Drho:       %.5f"
              % (r_q2n, r_ergas, r_sam, r_q, D_rho))

        if show_results_flag:
            from show_results import show
            show(I_MS, I_PAN, outputs, ratio, "Outcomes")

        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Z-PNN',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Package for Full-Resolution quality assessment for pansharpening'
                                                 'It consists of Reprojected Metrics, trying to solve the coregistration'
                                                 'problem and a new spatial no-reference metric.',
                                     epilog='''\
    Reference: 
    Full-resolution quality assessment for pansharpening
    G. Scarpa, M. Ciotola

    Authors: 
    Image Processing Research Group of University Federico II of Naples 
    ('GRIP-UNINA')
                                         '''
                                     )
    optional = parser._action_groups.pop()
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument("-i", "--input", type=str, required=True,
                               help='The path of the .mat file which contains the MS '
                                    'and PAN images. For more details, please refer '
                                    'to the GitHub documentation.')
    requiredNamed.add_argument('-f', '--fused', type=str, required=True, help='The path of pansharpened image.')

    requiredNamed.add_argument('-s', '--sensor', type=str, required=True, choices=["WV3", "WV2", 'GE1', "QB", "IKONOS"],
                               help='The sensor that has acquired the test image. Available sensors are '
                                    'WorldView-3 (WV3), WorldView-2 (WV2), GeoEye1 (GE1), QuickBird (QB), IKONOS')

    default_out_path = 'Outputs/'
    optional.add_argument("-o", "--out_dir", type=str, default=default_out_path,
                          help='The directory in which save the outcome.')

    optional.add_argument('-n_gpu', "--gpu_number", type=int, default=0, help='Number of the GPU on which perform the '
                                                                              'algorithm.')
    optional.add_argument("--use_cpu", action="store_true",
                          help='Force the system to use CPU instead of GPU. It could solve OOM problems, but the '
                               'algorithm will be slower.')
    optional.add_argument("--save_outputs", action="store_true",
                          help='Save the results in a .mat file. Please, use out_dir flag to indicate where to save.')

    optional.add_argument("--show_results", action="store_true", help="Enable the visualization of the outcomes.")
    optional.add_argument("--ratio", type=int, default=4, help='PAN-MS resolution ratio.')

    parser._action_groups.append(optional)
    arguments = parser.parse_args()

    main_metrics(arguments)