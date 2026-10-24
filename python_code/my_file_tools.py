"""
My useful file tools
"""

import matplotlib.pyplot as plt 
import os
from pathlib import Path
import re

def save_fig_as(path, filename):
    """
    Joins automatically the path and the desired filename to save the current figure

    Parameters
    -----
    path : string 
        Path to the directory in which the figure should be saved
    filename : string
        Desired filename for the saved figure
    
    Returns 
    -----
    Nothing
    """
    figure_filename = os.path.join(str(path), str(filename))
    plt.savefig(figure_filename)


def get_parameter_regexp(filename, parameter_to_fetch):
    """
    Using regular expression, fetch the value of a parameter used in the simulation from the logfile
    Might be overkill, but it should withstand changes in the logfile layout.

    Parameters
    -----
    filename : String
        Name of the logfile
    parameter_to_fetch : String
        Parameter to look for in the logfile 
    
    Returns
    -----
    parameter_value : float
        Value of the parameter fetched in the simulation logfile
    """
    # Avoid surprise
    filename = str(filename)
    parameter_to_fetch = str(parameter_to_fetch)

    # [0-9]+ : search for one or more numeric character
    # i.e. "243" : finds 243 ; 245a6 : finds 245 ; b24r : finds 24
    pattern = "{} = [+-]?([0-9]*[.])?[0-9]+".format(parameter_to_fetch)
    regexp = re.compile(pattern)

    # TODO : there should be a try/except in case the parameter is not found
    with open(filename,'r') as f:
        for line in f:
            result = regexp.search(line)
            if not result == None:
                # Only interested in the first match
                pattern_found = result.group(0)
                break

    # Process the result
    try:
        pattern_found = pattern_found.split()         
    except NameError:
        print("No pattern was found")
    else:
        # Get the last item of the list (which should be the number of points) and cast into a float by default
        # Manually recast it if needed
        parameter_value = float(pattern_found[-1])

        return parameter_value
