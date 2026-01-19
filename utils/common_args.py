import argparse



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')
def add_feature_preprocess_args(parser):
   
    parser.add_argument('--pt_type', type=str, default='tukey',
                        choices=['tukey', 'boxcox', 'none'],
                        help='Feature Power Conversion Type：Tukey / BoxCox / None')

    parser.add_argument('--pt_lam', type=float, default=0.5,
                        help='The λ hyperparameter of the power transform (default 0.5)')

    parser.add_argument('--pt_shift', type=float, default=1e-5,
                        help='After nonnegation, an additional translation is performed to avoid BoxCox/Tukey encountering 0 or negative numbers.')
    return parser

def add_map_args(parser):
   
    
    parser.add_argument(
        '--use_map', type=str2bool, default=True,
        help='[MAP] Should MAP prior aggregation be enabled during the episodic testing phase?'
    )

    
    parser.add_argument(
        '--map_estimate_cov', type=str2bool, default=False,
        help=''
    )

   
    parser.add_argument(
        '--map_use_semantic', type=str2bool, default=True,
        help='[MAP] Choose between using semantic space to select the top-k base class (True) or visual feature space (False).'
    )

    # ===== alpha / beta / topk / cov_bias =====
    parser.add_argument(
        '--map_alpha', type=float, default=0.33,
        help='[MAP] The mean MAP fusion coefficient alpha (the larger the value, the more it is biased towards the support set statistics).'
    )
    parser.add_argument(
        '--map_beta', type=float, default=0.0,
        help='[MAP] The MAP fusion coefficient beta of covariance (the larger the value, the more it is biased towards the covariance of the support set).'
    )
    parser.add_argument(
        '--map_topk', type=int, default=9,
        help='[MAP] The top-k nearest neighbors selected from the base class are used to construct the prior.'
    )
    parser.add_argument(
        '--map_cov_bias', type=float, default=0.0,
        help='[MAP] Numerical stability bias added to the covariance diagonal cov_bias     default=0.05'
    )
    parser.add_argument('--map_tau', type=float, default=1.0, help='Temperature for similarity-weighted MAP')

    return parser
