import os
import random
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Submit FEW configuration comaprisons '
                                     'to the cluster.', add_help=False)

    parser.add_argument('-est', type=str, dest='est', help='Comma-separated list of '
                        'python files containing regression model configurations',
                        default='dtr,lasso')
                        # choices = ['lasso','svr','lsvr','rfr','dtr','knr'])

    parser.add_argument('-clf', type=str, dest = 'clf', help='Comma-separated list of '
                        'python files containing classification model configurations',
                        default='lr,dtc')
                        # choices = ['lr','svc','rfc','dtc','dc','knc','sgd'])

    parser.add_argument('-save_dir', type=str, default='/home/lacava/results/',
                        help='result path')

    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')

    parser.add_argument('-reg_probs', action='store', dest='reg_problems', type=str,
                        default='concrete,enc,housing,uball5d,yacht',
                        help='Comma-separated list of regression problems.')

    parser.add_argument('-clf_probs', action='store', dest='clf_problems', type=str,
                        default='heart,bc_clean,yeast,seg,wav',
                        help='Comma-separated list of classification problems.')

    parser.add_argument('-flags', action='store', dest='flags', type=str,
                        default='weight_parents',
                        help='Comma-separated list of flags to turn on/off '
                        'in comparison.')

    parser.add_argument('-vals', action='store',dest ='vals',default=None,type=str,
                        help='Specify values to compare in range as follows: '
                        'name:val1,val2,val3')

    parser.add_argument('-data_dir', action='store',type =str, dest ='data_dir',
                        default='/home/lacava/data/', help='Path to datafiles')

    parser.add_argument('-n_cores', action='store',type =int, dest ='n_cores',
                        default=1, help='Number of cores per job')

    parser.add_argument('-n_trials', action='store',type =int, dest ='n_trials',
                        default=10, help='Number of repeate trials of each job')

    args = parser.parse_args()
    print(70*'=')
    # set values
    regressors = args.est.split(',')
    classifiers = args.clf.split(',')
    reg_problems = [] if args.reg_problems=='' else args.reg_problems.split(',')
    clf_problems = [] if args.clf_problems=='' else args.clf_problems.split(',')
    flags = args.flags.split(',')
    if flags:
        flags = ['--'+f for f in flags]
        flags += ['']
    print('flags: ',flags)
    if args.vals:
        vals = (args.vals.split(':')[0],args.vals.split(':')[1])
    n_cores = args.n_cores
    n_trials = args.n_trials
    data_dir = args.data_dir
    results_path = args.save_dir

    save_file = (results_path + 'FEW-config-compare-' + '-'.join(flags[:-1]).replace('--','')
                + '.csv')
    print('save_file:',save_file)
    # write header
    with open(save_file,'w') as out:
        out.write('dataset\tmethod\ttrial\tml\tscore\ttime\n')

    ################################################################## regression problems
    for p in reg_problems:
        print(p)
        dataset_name = data_dir + p + '.csv'

        for ml in regressors:
            print('\t','FEW-' + ml)
            for flag in flags:
                # print('flag: ',flag)
                method = flag.replace('--', '')
                if flag=='':
                    method = 'control'

                for i in range(n_trials):
                    job_name = 'FEW-' + ml + '_' + p + '_' + str(i)
                    # save_file = results_path + 'FEW-' + ml + '_' + p + '.csv'
                    out_file = results_path + '{JOB_NAME}_%J.out'.format(JOB_NAME=job_name)
                    error_file = out_file.split('.')[0] + '.err'

                    ################################################# submit job with flag
                    bjob_line = ('bsub -o {OUT_FILE} -e {ERROR_FILE} -n {N_CORES} '
                                 '-J {JOB_NAME} -R "span[hosts=1]" '
                                 '"python FEW_job.py {DATASET} {SAVE_FILE} {TRIAL} '
                                 '-method {METHOD} -ml {ML} {FLAG}"'.format(
                                                                    OUT_FILE=out_file,
                                                                    ERROR_FILE=error_file,
                                                                    N_CORES=n_cores,
                                                                    JOB_NAME=job_name,
                                                                    DATASET=dataset_name,
                                                                    SAVE_FILE=save_file,
                                                                    TRIAL=i,
                                                                    METHOD=method,
                                                                    ML=ml,
                                                                    FLAG=flag))
                    # print(bjob_line)
                    os.system(bjob_line)

    ############################################################## classification problems
    for p in clf_problems:
        print(p)
        dataset_name = data_dir + p + '.csv'

        for ml in classifiers:
            print('\t',ml)
            for flag in flags:
                # print('flag: ',flag)
                method = flag.replace('--', '')
                if flag=='':
                    method = 'control'

                for i in range(n_trials):
                    job_name = 'FEW-' + ml + '_' + p + '_' + str(i)
                    # save_file = results_path + 'FEW-' + ml + '_' + p + '.csv'
                    out_file = results_path + '{JOB_NAME}_%J.out'.format(JOB_NAME=job_name)
                    error_file = out_file.split('.')[0] + '.err'

                    ################################################# submit job with flag
                    bjob_line = ('bsub -o {OUT_FILE} -e {ERROR_FILE} -n {N_CORES} '
                                 '-J {JOB_NAME} -R "span[hosts=1]" '
                                 '"python FEW_job.py {DATASET} {SAVE_FILE} {TRIAL} '
                                 '-method {METHOD} -ml {ML} {FLAG} --class"'.format(
                                                                    OUT_FILE=out_file,
                                                                    ERROR_FILE=error_file,
                                                                    N_CORES=n_cores,
                                                                    JOB_NAME=job_name,
                                                                    DATASET=dataset_name,
                                                                    SAVE_FILE=save_file,
                                                                    TRIAL=i,
                                                                    METHOD=method,
                                                                    ML=ml,
                                                                    FLAG=flag))
                    # print(bjob_line)
                    os.system(bjob_line)

if __name__ == '__main__':
    main()
