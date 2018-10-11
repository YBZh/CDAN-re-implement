import argparse


def opts():
    parser = argparse.ArgumentParser(description='Train resnet on the cub dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path_source', type=str, default='/data/',
                        help='Root of the data set')
    parser.add_argument('--data_path_source_t', type=str, default='/data/',
                        help='Root of the data set')
    parser.add_argument('--data_path_target', type=str, default='/data/',
                        help='Root of the data set')
    parser.add_argument('--src', type=str, default='', help='The source data')
    parser.add_argument('--src_t', type=str, default='', help='The source data')
    parser.add_argument('--tar', type=str, default='', help='The target data')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=161, help='Number of epochs to train')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.0001, help='Weight decay (L2 penalty).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # checkpoints
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', type=str, default='', help='Checkpoints path to resume(default none)')
    parser.add_argument('--test_only', '-t', action='store_true', help='Test only flag')
    parser.add_argument('--calculate_percent_only', action='store_true', help='Test only flag')
    # Architecture
    parser.add_argument('--arch', type=str, default='alexnet', help='Model name')
    parser.add_argument('--lrplan', type=str, default='dao', help='step or dao')
    parser.add_argument('--domain_feature', type=str, default='original', help='the calculation of domain feature：original | full_bilinear')
    parser.add_argument('--adv_loss', type=str, default='reverse',
                        help='the type of adv loss: reverse | uniform | gan')
    parser.add_argument('--pretrained_checkpoint', type=str, default='', help='the source data pre-trained model')
    parser.add_argument('--pretrained', action='store_true', help='whether using pretrained model')
    parser.add_argument('--pretrained_fc', action='store_true', help='whether using pretrained model')

    parser.add_argument('--numclass_da', type=int, default=2, help='class Number of new model to be trained or fine-tuned')
    parser.add_argument('--numclass_s', type=int, default=31, help='class Number of new model to be trained or fine-tuned')
    # i/o
    parser.add_argument('--log', type=str, default='./checkpoints', help='Log folder')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print_freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--domain_freq', default=50, type=int, help='print frequency (default: 10)')
    args = parser.parse_args()
    args.log = args.log + '_' + args.src + str(2) + args.tar + '_' + args.arch + '_' + args.domain_feature + '_' + args.adv_loss

    return args
