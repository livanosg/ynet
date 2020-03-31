import argparse


PARSER = argparse.ArgumentParser(description='Train a model according to given hyperparameters.')
# Mode
PARSER.add_argument('-M', '--mode', type=str, default='train-and-eval', choices=['train', 'eval', 'pred', 'train-and-eval', 'test', 'lr'],  # TODO EXPORT MODEL
                    help='Define the estimator mode')
PARSER.add_argument('-nodist', action='store_false', default=True, help='Set distribution mode.')
PARSER.add_argument('-sd', '--seed', type=int, default=None, help='Random seed.')

# Model options
PARSER.add_argument('-load', '--load_model', type=str, default='', help=' Model folder to load.')
PARSER.add_argument('-resume', action='store_true', default=False, help='Continue training from loaded model.')
PARSER.add_argument('-brnch', '--branch', type=int, default=1, choices=[1, 2], help='Branch to train.')
PARSER.add_argument('-dr', '--dropout', type=float, default=0.5, help='Dropout rate.')
PARSER.add_argument('-cls', '--classes', type=int, default=2, choices=[2], help='Choose classes')
# Training dataset options
PARSER.add_argument('-modal', '--modality', type=str, default='CT', choices=['CT', 'MR', 'ALL'], help='Set type of training data.')
PARSER.add_argument('-augp', '--augm_prob', type=float, default=0.5, help='Probability for augmented image.')
PARSER.add_argument('-batch', '--batch_size', type=int, default=2, help='Mini-batch size.')
# Training hyper-parameters
PARSER.add_argument('-lr', '--lr', type=float, default=0.000008, help='Learning Rate.')
PARSER.add_argument('-dc', '--decays_per_train', type=int, default=1, help='Number of learning rate decays in a training session.')
PARSER.add_argument('-dcr', '--decay_rate', type=float, default=0.1, help='Decay rate for learning rate.')
PARSER.add_argument('-e', '--epochs', type=int, default=500, help='Training epochs.')
PARSER.add_argument('-es', '--early_stop', type=int, default=35, help='Epochs without minimizing target.')
# Estimator configuration
ARGS = PARSER.parse_args()

if __name__ == '__main__':
    # print(ARGS.nodist)
    from train import estimator_mod
    estimator_mod(ARGS)
