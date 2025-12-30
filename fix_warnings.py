import os

files = [f for f in os.listdir('.') if f.startswith('exp') and f.endswith('.py')]

header_code = [
    "import os\n",
    "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n", # 3 = FATAL only, 2 = ERROR only. 3 is safer to hide "optimized" msg
    "os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n"
]

for file in files:
    with open(file, 'r') as f:
        lines = f.readlines()
    
    # Avoid double patching headers
    if "TF_CPP_MIN_LOG_LEVEL" in "".join(lines[:10]):
        print(f"Skipping header patch for {file}")
        continue

    # Prepend header
    new_lines = header_code + lines
    
    with open(file, 'w') as f:
        f.writelines(new_lines)

print(f"Patched {len(files)} files with warning suppressions.")
