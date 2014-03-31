def agts(queue):
    queue.add('run.py .', ncpus=8, walltime=40 * 60)
    queue.add('run.py .', ncpus=8, walltime=40 * 60)
    queue.add('run.py .', ncpus=4, walltime=10 * 60)
    queue.add('run.py .', ncpus=4, walltime=10 * 60)
    queue.add('run.py .', ncpus=1, walltime=30)

