import subprocess

# Function to run the scripts
def run_script(script_name):
    print(f"Running {script_name}...")
    result = subprocess.run(['python3', script_name], capture_output=True, text=True)
    print(result.stdout)  # Print the output
    if result.stderr:
        print(f"Error: {result.stderr}")  # Print any error

# List of scripts to run
scripts = ["script_1.py", "script_2.py", "script_3.py", "script_4.py"]

# Run all scripts
for script in scripts:
    run_script(script)
