
for file_i in range(20):

    with open('Runs_metast/Metast_run{}.txt'.format(file_i+1), 'r') as f:
        lines = f.readlines()

    output = []
    for line in lines:
        words = line.split()
        for i, word in enumerate(words):
            if '<=' in word:
                output.append(words[i-1])
                break

    with open('Results_metast/output_Metast_run{}.txt'.format(file_i+1), 'w') as f:
        f.write('\n'.join(output))
