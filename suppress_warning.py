import subprocess
import sys
import os

# usage: py -3.14 suppress_warning.py py -3.13 script.py

if len(sys.argv) < 2:
    print("Usage: suppress_warning.py <command>")
    sys.exit(1)

command = sys.argv[1:]

# Run the command
result = subprocess.run(command, capture_output=True, text=True)

# Print stdout as is
print(result.stdout, end='')

# Filter and print stderr
stderr_lines = result.stderr.splitlines()
for line in stderr_lines:
    if "Could not find platform independent libraries" in line or "<prefix>" in line:
        continue
    print(line, file=sys.stderr)

sys.exit(result.returncode)
