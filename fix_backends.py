import os

files = [f for f in os.listdir('.') if f.startswith('exp') and f.endswith('.py')]

for file in files:
    with open(file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    patched = False
    for line in lines:
        if 'import matplotlib.pyplot as plt' in line and not patched:
            new_lines.append("import matplotlib\n")
            new_lines.append("matplotlib.use('Agg')\n")
            new_lines.append(line)
            patched = True
        else:
            new_lines.append(line)
            
    with open(file, 'w') as f:
        f.writelines(new_lines)

print(f"Patched {len(files)} files.")
