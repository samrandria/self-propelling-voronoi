"""
Test for calling a script with arguments
and using shell commands
"""

import sys
import subprocess
import shlex

if __name__ == '__main__':
    a = int(sys.argv[1])
    b = float(sys.argv[2])
    print(a)
    print(b)
    print(type(sys.argv[1]))
    print(type(sys.argv[2]))
    print(type(a))
    print(type(b))
    subprocess.run("echo Hello World", shell=True)
    # subprocess.run(["dir"])
    # Attention aux shell=True !!!!
    subprocess.run("ls -l test_case_shell*", shell=True)
    command = "ls -l *.py"
    print(command)
    
    # Should use the shlex.quote() function to take care of proper escaping.
    # And prevent code injection
    shlexcommand = shlex.quote(command)
    print(shlexcommand)