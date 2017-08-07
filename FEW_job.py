import multiprocessing

# main functions
def main():
    """Main function that is called when FEW is run on the command line"""
    parser = argparse.ArgumentParser(description='A feature engineering wrapper'
                                     ' for machine learning algorithms.',
                                     add_help=False)

    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to run FEW on; ensure that the '
                        'target/label column is labeled as "label" or "class".')

    parser.add_argument('OUTPUT_FILE', type=str, help='File to export results.')

    parser.add_argument('TRIAL', action='store',type=str,
                        default='0', help='Trial number')

    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')

    parser.add_argument('-is', action='store',dest='INPUT_SEPARATOR',
                        default=None,type=str,
                        help='Character separating columns in the input file.')

    parser.add_argument('-g', action='store', dest='GENERATIONS', default=100,
                        type=int,
                        help='Number of generations to run FEW.')

    parser.add_argument('-p', action='store', dest='POPULATION_SIZE',default=50,
                         help='Number of individuals in the GP population. '
                         'Follow the number with x to set population size as a'
                         'multiple of raw feature size.')

    parser.add_argument('-mr', action='store', dest='MUTATION_RATE',default=0.5,
                        type=float,
                        help='GP mutation rate in the range [0.0, 1.0].')

    parser.add_argument('-xr', action='store', dest='CROSSOVER_RATE',
                        default=0.5,type=float,
                        help='GP crossover rate in the range [0.0, 1.0].')

    parser.add_argument('-ml', action='store', dest='MACHINE_LEARNER',
                        default=None,
                        choices = ['lasso','svr','lsvr','lr','svc','rfc','rfr',
                                   'dtc','dtr','dc','knc','knr','sgd'],
                        type=str, help='ML algorithm to pair with features. '
                        'Default: Lasso (regression), LogisticRegression '
                        '(classification)')

    parser.add_argument('-min_depth', action='store', dest='MIN_DEPTH',
                        default=1,type=int,
                        help='Minimum length of GP programs.')

    parser.add_argument('-max_depth', action='store', dest='MAX_DEPTH',
                        default=2,type=int,
                        help='Maximum number of nodes in GP programs.')

    parser.add_argument('-max_depth_init', action='store',dest='MAX_DEPTH_INIT',
                        default=2,type=int,
                        help='Maximum nodes in initial programs.')

    parser.add_argument('-op_weight', action='store',dest='OP_WEIGHT',default=1,
                        type=bool, help='Weight attributes for incuded in'
                        ' features based on ML scores. Default: off')

    parser.add_argument('-ms', action='store', dest='MAX_STALL',default=100,
                        type=int, help='If model CV does not '
                        'improve for this many generations, end optimization.')

    parser.add_argument('--weight_parents', action='store_true',
                        dest='WEIGHT_PARENTS',default=True,
                        help='Feature importance weights parent selection.')

    parser.add_argument('--lex_size', action='store_true',dest='LEX_SIZE',default=False,
                        help='Size mediated parent selection for lexicase survival.')

    parser.add_argument('-sel', action='store', dest='SEL',
                        default='epsilon_lexicase',
                        choices = ['tournament','lexicase','epsilon_lexicase',
                                   'deterministic_crowding','random'],
                        type=str, help='Selection method (Default: tournament)')

    parser.add_argument('-tourn_size', action='store', dest='TOURN_SIZE',
                        default=2, type=int,
                        help='Tournament size (Default: 2)')

    parser.add_argument('-fit', action='store', dest='FIT_CHOICE', default=None,
                        choices = ['mse','mae','r2','vaf','mse_rel','mae_rel',
                                   'r2_rel','vaf_rel','silhouette','inertia',
                                   'separation','fisher','random','relief'],
                        type=str,
                        help='Fitness metric (Default: dependent on ml used)')

    parser.add_argument('--no_seed', action='store_false', dest='SEED_WITH_ML',
                        default=True,
                        help='Turn off initial GP population seeding.')

    parser.add_argument('--elitism', action='store_true', dest='ELITISM',
                        default=False,
                        help='Force survival of best feature in GP population.')

    parser.add_argument('--erc', action='store_true', dest='ERC', default=False,
                    help='Use random constants in GP feature construction.')

    parser.add_argument('--bool', action='store_true', dest='BOOLEAN',
                        default=False,
                        help='Include boolean operators in features.')

    parser.add_argument('-otype', action='store', dest='OTYPE', default='f',
                        choices=['f','b'],
                        type=str,
                        help='Feature output type. f: float, b: boolean.')

    parser.add_argument('-ops', action='store', dest='OPS', default=None,
                        type=str,
                        help='Specify operators separated by commas')

    parser.add_argument('--class', action='store_true', dest='CLASSIFICATION',
                        default=False,
                        help='Conduct classification rather than regression.')

    parser.add_argument('--mdr', action='store_true',dest='MDR',default=False,
                        help='Use MDR nodes.')

    parser.add_argument('--diversity', action='store_true',
                        dest='TRACK_DIVERSITY', default=False,
                        help='Store diversity of feature transforms each gen.')

    parser.add_argument('--clean', action='store_true', dest='CLEAN',
                        default=False,
                        help='Clean input data of missing values.')

    parser.add_argument('--no_lib', action='store_false', dest='c',
                        default=True,
                        help='Don''t use optimized c libraries.')

    parser.add_argument('-s', action='store', dest='RANDOM_STATE',
                        default=None,
                        type=int,
                        help='Random number generator seed for reproducibility.'
                        'Note that using multi-threading may make exact results'
                        ' impossible to reproduce.')

    parser.add_argument('-v', action='store', dest='VERBOSITY', default=1,
                        choices=[0, 1, 2, 3], type=int,
                        help='How much information FEW communicates while it is'
                        ' running: 0 = none, 1 = minimal, 2 = lots, 3 = all.')

    parser.add_argument('--no-update-check', action='store_true',
                        dest='DISABLE_UPDATE_CHECK', default=False,
                        help='Don''t check the FEW version.')

    parser.add_argument('-method', action='store', dest='METHOD', type=str,
                        help="method name to store in output")

    args = parser.parse_args()

    # if args.VERBOSITY >= 2:
    #     print('\nFEW settings:')
    #     for arg in sorted(args.__dict__):
    #         if arg == 'DISABLE_UPDATE_CHECK':
    #             continue
    #         print('{}\t=\t{}'.format(arg, args.__dict__[arg]))
    #     print('')

    # load data from csv file
    if args.INPUT_SEPARATOR is None:
        input_data = pd.read_csv(args.INPUT_FILE, sep=args.INPUT_SEPARATOR,
                                 engine='python')
    else: # use c engine for read_csv is separator is specified
        input_data = pd.read_csv(args.INPUT_FILE, sep=args.INPUT_SEPARATOR)

    # if 'Label' in input_data.columns.values:
    input_data.rename(columns={'Label': 'label','Class':'label','class':'label',
                               'target':'label'}, inplace=True)

    RANDOM_STATE = args.RANDOM_STATE

    X = input_data.drop('label', axis=1).values
    y = input_data['label'].values

    learner = FEW(generations=args.GENERATIONS,
                  population_size=args.POPULATION_SIZE,
                  mutation_rate=args.MUTATION_RATE,
                  crossover_rate=args.CROSSOVER_RATE,
                  ml = ml_dict[args.MACHINE_LEARNER],
                  min_depth = args.MIN_DEPTH,max_depth = args.MAX_DEPTH,
                  sel = args.SEL, tourn_size = args.TOURN_SIZE,
                  seed_with_ml = args.SEED_WITH_ML, op_weight = args.OP_WEIGHT,
                  max_stall = args.MAX_STALL,
                  erc = args.ERC, random_state=args.RANDOM_STATE,
                  verbosity=args.VERBOSITY,
                  disable_update_check=args.DISABLE_UPDATE_CHECK,
                  fit_choice = args.FIT_CHOICE,boolean=args.BOOLEAN,
                  classification=args.CLASSIFICATION,clean = args.CLEAN,
                  track_diversity=args.TRACK_DIVERSITY,mdr=args.MDR,
                  otype=args.OTYPE,c=args.c, lex_size = args.LEX_SIZE,
                  weight_parents = args.WEIGHT_PARENTS,operators=args.OPS)

    t0 = time.time()
    scores = cross_val_score(learner,X,y)
    runtime = time.time() - t0

    # print results
    print('dataset\tmethod\ttrial\tml\tscore\ttime')
    out_text = '\t'.join([args.INPUT_FILE.split('/')[-1].split('.')[0],
                          args.METHOD,
                          args.TRIAL,
                          str(learner.ml_type),
                          str(np.mean(scores)),
                          str(runtime)])
    # WIP: add printout of pareto archive
    # print summary results
    with open(args.OUTPUT_FILE,'a') as out:
        out.write(out_text+'\n')
    print(out_text)
    sys.stdout.flush()

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    import numpy as np
    from few import FEW
    from sklearn.model_selection import cross_val_score
    import time
    import argparse
    import pandas as pd
    import sys
    from sklearn.linear_model import LassoLarsCV, LogisticRegression, SGDClassifier
    from sklearn.svm import SVR, LinearSVR, SVC, LinearSVC
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from DistanceClassifier import DistanceClassifier

    # dictionary of ml options
    ml_dict = {
            'lasso': LassoLarsCV(),
            'svr': SVR(),
            'lsvr': LinearSVR(),
            'lr': LogisticRegression(solver='sag'),
            'sgd': SGDClassifier(loss='log',penalty='l1'),
            'svc': SVC(),
            'lsvc': LinearSVC(),
            'rfc': RandomForestClassifier(),
            'rfr': RandomForestRegressor(),
            'dtc': DecisionTreeClassifier(),
            'dtr': DecisionTreeRegressor(),
            'dc': DistanceClassifier(),
            'knc': KNeighborsClassifier(),
            'knr': KNeighborsRegressor(),
            None: None
    }

    main()
