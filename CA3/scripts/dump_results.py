
import os

root_dir = './logs'

for d in sorted(os.listdir(root_dir)):
    if not d.endswith('.log'):
        continue
    path = os.path.join(root_dir, d)
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    results = '\n'.join([l for l in lines if l.startswith('Valid') or l.startswith('Best')])
    if len(results) > 0:
        print(f'\n{path}:\n{results}')
