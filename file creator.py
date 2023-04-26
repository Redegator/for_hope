

for file_i in range(20):

    with open('Runs_metast/Metast_run{}.txt'.format(file_i+1), 'w') as f:
        f.write(f"Run {file_i+1} \n")