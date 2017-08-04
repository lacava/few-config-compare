import os
import random
import sys

def main():
    parser = argparse.ArgumentParser(description='combine csv files in folder into one csv file.',
                                     add_help=False)

    parser.add_argument('-est', type=str, dest='est', help='Comma-separated list of '
                        'python files containing regression model configurations',
                        default='FEW-lasso,FEW-dtr')
    parser.add_argument('-clf', type=str, dest = 'clf', help='Comma-separated list of '
                        'python files containing classification model configurations',
                        default='FEW-log,FEW-dtc')
    parser.add_argument('-save_dir', type=str, default='/home/lacava/results/', help='result file')

    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    parser.add_argument('-reg_probs', action='store', dest='reg_problems', type=str,
                        default='concrete,enc,housing,uball5d,yacht',
                        type=str, help='Comma-separated list of regression problems.')
    parser.add_argument('-class_probs', action='store', dest='class_problems', type=str,
                        default='heart,bc_clean,yeast,seg,wav',
                        type=str, help='Comma-separated list of classification problems.')

    parser.add_argument('-flags', action='store', dest='flags', type=str,
                        default='weight_parents',
                        type=str, help='Comma-separated list of flags to turn on/off '
                        'in comparison.')

    parser.add_argument('-vals', action='store',dest ='vals',default=False,
                        help='Specify values to compare in range as follows: '
                        'name:val1,val2,val3')

    parser.add_argument('-data_dir', action='store',type =str, dest ='data_dir',
                        default='/home/lacava/data/', help='Path to datafiles')

    parser.add_argument('-n_cores', action='store',type =int, dest ='n_cores',
                        default=1, help='Number of cores per job')

    parser.add_argument('-n_trials', action='store',type =int, dest ='n_trials',
                        default=1, help='Number of repeate trials of each job')

    args = parser.parse_args()

    # set values
    estimators = ','.split(args.est)
    classifiers = ','.split(args.clf)
    reg_problems = ','.split(args.reg_problems)
    clf_problems = ','.split(args.clf_problems)
    flags = ','.split(args.flags)
    #vals = ':'.split[0]:':'.split[1].
    n_cores = args.n_trials
    n_trials = args.n_trials
    data_dir = args.data_dir
    results_path = args.save_dir

    for p in problems:
        print(p)
        # dataset = fetch_data(p,local_cache_dir=data_dir)
    # for dataset in glob('/home/lacava/data/regression/*.txt'):
        dataset_name = data_dir + p + '.csv'
        results_path = '/home/lacava/results/maize/'

        for ml in estimators:
            print('\t',ml)
            for i in range(n_trials):

                job_name = ml + '_' + p + '_' + str(i)
                save_file = results_path + ml + '_' + p + '.csv'
                out_file = results_path + '{JOB_NAME}_%J.out'.format(JOB_NAME=job_name)
                error_file = out_file.split('.')[0] + '.err'
                #write header
                with open(save_file,'w') as out:
                    out.write('dataset\tmethod\ttrial\tparameters\tr2\ttime\n')
                #submit job
                bjob_line = ('bsub -o {OUT_FILE} -e {ERROR_FILE} -n {N_CORES} '
                             '-J {JOB_NAME} -R "span[hosts=1]" '
                             '"python {ML}.py {DATASET} {SAVE_FILE} {TRIAL} '
                             '{N_CORES}"'.format(OUT_FILE=out_file,
                                                 ERROR_FILE=error_file,
                                                 JOB_NAME=job_name,
                                                 N_CORES=n_cores,
                                                 ML=ml,
                                                 DATASET=dataset_name,
                                                 SAVE_FILE=save_file,
                                                 TRIAL=i,
                                                 NCORES=n_cores))
                # print(bjob_line)
                os.system(bjob_line)
