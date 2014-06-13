def agts(queue):
    queue.add('bandstructure.py', ncpus=1, walltime=30, creates=['bandstructure.png'])
    
