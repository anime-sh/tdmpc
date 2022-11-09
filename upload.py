import os 
import subprocess
with open('tasks.txt') as f:
    
    pref=os.getcwd()
    for line in f:
        line=line.strip()
        path=f'{pref}/logs/{line}/state/default/'
        subd=[f'{path}{name}' for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        for subp in subd:
            os.chdir(subp)
            print(os.getcwd())
            bashCommand = "wandb sync --sync-all"
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(output)
            print(error)