import os
from collections import OrderedDict 
import subprocess

def save_experiment_config(args, experiment_dir):
    logfile = os.path.join(experiment_dir, 'parameters.txt')
    log_file = open(logfile, 'w')
    try: 
        lines = subprocess.check_output( 
            ['git', 'log'], stderr=subprocess.STDOUT 
        ).decode("utf-8").split("\n") 
        commit = lines[0].split(' ')[1]
    except Expections as e: 
        commit = '###Not a git repo###'
    p = OrderedDict()
    p['code_version'] = commit
    p['dataset'] = args.dataset
    p['basenet'] = args.basenet
    p['batch_size'] = args.batch_size
    p['resume'] = args.resume
    p['start_iter'] = args.start_iter
    p['num_workers'] = args.num_workers
    p['cuda'] = args.cuda
    p['lr'] = args.lr
    p['momentum'] = args.momentum
    p['weight_decay'] = args.weight_decay
    p['gamma'] = args.gamma
    p['save_folder'] = args.save_folder

    for key, val in p.items():
        log_file.write(key + ': ' + str(val) + '\n')
    log_file.close()