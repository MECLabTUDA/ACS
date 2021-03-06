import argparse
import torch

# parse train options
def _get_parser():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--experiment-name', type=str, default='', help='experiment name for new or resume')
    parser.add_argument('--nr-runs', type=int, default=1, help='# of runs')

    # hardware
    parser.add_argument('--device', type=str, default='cuda', help='device type cpu or cuda')
    parser.add_argument("--device-ids", nargs="+", default=[0], type=int, help="ID(s) of GPU device(s)")
    parser.add_argument('--n-workers', type=int, default=2, help='# multiplied by # of GPU to get # of total workers')

    # dataset
    parser.add_argument('--test-ratio', type=float, default=0.2, help='ratio of data to be used for testing')
    parser.add_argument('--val-ratio', type=float, default=0.125, help='ratio of data to be used for validation')
    parser.add_argument('--input_dim_c', type=int, default=1, help='input channels for images') 
    parser.add_argument('--input_dim_hw', type=int, default=256, help='height and width for images')
    parser.add_argument('--no-resize', action='store_true', help='specify if images should not be resized')
    parser.add_argument('--augmentation', type=str, default='none', help='augmentation to be used')
    parser.add_argument('--n-samples', type=int, default=None, help='# of samples per dataloader, only use when debugging')
    parser.add_argument('--sampler', action='store_true', help='sample datasets to have equal # of samples')
    parser.add_argument('--combination', type=int, default=0, help='0: ab->c, 1: ac->b, 2: bc->a')

    # training
    parser.add_argument('--epochs', type=int, default=60, help='# of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for training stage 1')
    parser.add_argument('--lr-2', type=float, default=1e-4, help='learning rate for training stage 2')
    parser.add_argument('--domain-code-size', type=int, default=3, help='# of domains present in training stage 1')
    parser.add_argument('--cross-validation', action='store_true', help='specify if cross validation should be used')
    parser.add_argument('--d-iter', type=int, default=1, help='discriminator update iterations per epoch')
    
    # resume training
    parser.add_argument('--resume-epoch', type=int, default=None, help='resume training at epoch, -1 for latest, select run using experiment-name argument')
    
    # logging
    parser.add_argument('--eval-interval', type=int, default=7, help='evaluation interval (on all datasets)')
    parser.add_argument('--save-interval', type=int, default=1, help='save interval')
    parser.add_argument('--display-interval', type=int, default=1, help='display/tensorboard interval')
    
    # evaluation
    parser.add_argument('--eval', action='store_true', help='for general evaluation (only eval on test)')
    parser.add_argument('--lambda-eval', action='store_true', help='for tuning lambda (only eval on val)')

    # loss weighting
    parser.add_argument('--lambda-vae', type=float, default=5, help='lambda tuning vae loss')
    parser.add_argument('--lambda-c-adv', type=float, default=1, help='lambda tuning content adversarial loss')
    parser.add_argument('--lambda-lcr', type=float, default=1e-4, help='lambda tuning lcr loss')
    parser.add_argument('--lambda-seg', type=float, default=5, help='lambda tuning segmentation loss')
    parser.add_argument('--lambda-c-recon', type=float, default=0, help='lambda tuning content reconstruction loss')
    parser.add_argument('--lambda-gan', type=float, default=5, help='lambda tuning gan loss')
    parser.add_argument('--lambda-d', type=float, default=1, help='lambda for tuning MAS or KD loss')

    # U-Net
    parser.add_argument('--unet-only', action='store_true', help='only train UNet')
    parser.add_argument('--unet-dropout', type=float, default=0, help='apply dropout to UNet')
    parser.add_argument('--unet-monte-carlo-dropout', type=float, default=0, help='apply monte carlo dropout to UNet')
    parser.add_argument('--unet-preactivation', action='store_true', help='UNet preactivation; True: norm, act, conv; False:conv, norm, act')

    return parser

def parse_args(argv):
    """Parses arguments passed from the console as, e.g.
    'python ptt/main.py --epochs 3' """

    parser = _get_parser()
    args = parser.parse_args(argv)
    
    args.device = str(args.device+':'+str(args.device_ids[0]) if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    device_name = str(torch.cuda.get_device_name(args.device) if args.device == "cuda" else args.device)
    print('Device name: {}'.format(device_name))
    args.input_shape = (args.input_dim_c, args.input_dim_hw, args.input_dim_hw)

    # add dummy class
    args.domain_code_size = args.domain_code_size + 1

    return args

def parse_args_as_dict(argv):
    """Parses arguments passed from the console and returns a dictionary """
    return vars(parse_args(argv))

def parse_dict_as_args(dictionary):
    """Parses arguments given in a dictionary form"""
    argv = []
    for key, value in dictionary.items():
        if isinstance(value, bool):
            if value:
                argv.append('--'+key)
        else:
            argv.append('--'+key)
            argv.append(str(value))
    return parse_args(argv)