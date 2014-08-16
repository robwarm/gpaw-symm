def agts(queue):
    a = queue.add('dcdft_gpaw.py', ncpus=4, walltime=20)
    queue.add('testdb.py', deps=a, ncpus=1, walltime=5)
    queue.add('extract.py dcdft.db', deps=a, ncpus=1, walltime=5,
              creates='exercise_dcdft.db_raw.txt')

