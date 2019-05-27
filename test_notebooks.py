import argparse
import contextlib
import glob
import io
import math
import numbers
import os
import pickle
import platform
import re
import sys
import types

import colorama
import nbformat
import numpy as np
import pkg_resources
from nbconvert import PythonExporter


def print_warning(message):
    print(colorama.Fore.YELLOW + colorama.Style.BRIGHT + 'WARNING: ' + message + colorama.Style.RESET_ALL)


def check_var(stud_val, sol_val, var_name_style):
    def unify_vector(vec):
        # Remove single dimensions
        vec = np.squeeze(vec)

        # Convert column or row matrices to vectors
        if len(vec.shape) == 2 and (vec.shape[0] == 1 or vec.shape[1] == 1):
            return vec.reshape(-1)
        else:
            return vec

    if type(stud_val) != type(sol_val):
        print_warning('The variable %s has the type %s but the type %s is expected. Even though it is possible that the other checks still work, it can also lead to errors.' % (var_name_style, type(stud_val), type(sol_val)))

    if type(stud_val) == list or type(stud_val) == tuple:
        assert len(stud_val) == len(sol_val), 'The list %s does not have the correct length\nGot: %d\nExpected: %d' % (var_name_style, len(stud_val), len(sol_val))

        if any([isinstance(val, np.ndarray) or type(val) == list or type(val) == tuple for val in stud_val]):
            # Check nested lists recursively
            for v1, v2 in zip(stud_val, sol_val):
                check_var(v1, v2, var_name_style)
            return
        elif all([isinstance(val, numbers.Number) for val in stud_val]):
            # The list contains only numbers --> unify to numpy arrays
            stud_val = np.asarray(stud_val)
            sol_val = np.asarray(sol_val)
        else:
            assert stud_val == sol_val, 'The content of the list %s does not match with the solution\nGot: %s\nExpected: %s' % (var_name_style, stud_val, sol_val)
            return

    if isinstance(stud_val, np.ndarray):
        stud_val = unify_vector(stud_val)
        sol_val = unify_vector(sol_val)

        if stud_val.dtype != sol_val.dtype:
            print_warning('The type of your array %s has the type %s but the type %s is expected. Even though it is possible that the other checks still work, it can also lead to errors.' % (var_name_style, stud_val.dtype, sol_val.dtype))

        assert stud_val.shape == sol_val.shape, 'Your array %s does not have the correct dimensions\nGot: %s\nExpected: %s' % (var_name_style, stud_val.shape, sol_val.shape)

    if isinstance(stud_val, np.ndarray) or isinstance(stud_val, (np.floating, float)):
        assert np.allclose(stud_val, sol_val, rtol=1e-04, atol=1e-05), 'The content of the array %s does not match with the solution (tolerance considered)\nGot: %s\nExpected: %s' % (var_name_style, stud_val, sol_val)
    else:
        assert stud_val == sol_val, 'The variable %s is not set to the correct value\nGot: %s\nExpected: %s' % (var_name_style, stud_val, sol_val)


def check_vars(stud_vars, sol_vars):
    def function_name(function):
        # Returns the name of a function object
        match = re.search(r'function\s+([^\s]+)\s+at', str(function))
        if match:
            return match.group(1)
        else:
            return ''

    # Compare all variables
    error_messages = []
    for var_name in sol_vars.keys():
        if var_name == 'sol_vars_code':
            continue

        try:
            var_name_style = colorama.Style.NORMAL + var_name + colorama.Style.BRIGHT
            assert var_name in stud_vars, 'You do not have a variable with the name %s in your script.' % var_name_style

            stud_val = stud_vars[var_name]
            sol_val = sol_vars[var_name]

            if isinstance(stud_val, types.FunctionType):
                # For functions, compare only the names
                stud_val = function_name(stud_val)

            check_var(stud_val, sol_val, var_name_style)
        except AssertionError as failure:
            error_messages.append(colorama.Fore.RED + colorama.Style.BRIGHT + str(failure) + colorama.Style.RESET_ALL)
            continue

    # Compare all code snippets
    if 'sol_vars_code' in sol_vars:
        for var_code in sol_vars['sol_vars_code']:
            try:
                var_name_style = colorama.Style.NORMAL + var_code['name'] + colorama.Style.BRIGHT

                stud_val = eval(var_code['code'], stud_vars)    # Runs the code in the student's environment
                sol_val = var_code['result']

                check_var(stud_val, sol_val, var_name_style)
            except AssertionError as failure:
                error_messages.append(colorama.Fore.RED + colorama.Style.BRIGHT + str(failure) + '\nDescription: ' + var_code['description'] + '\nNote: This is a generated variable.' + colorama.Style.RESET_ALL)
                continue
            except Exception as error:
                msg = 'Could not execute the code ' + var_code['code'] + ' in the notebook.\nMessage: ' + str(error)
                error_messages.append(colorama.Fore.RED + colorama.Style.BRIGHT + msg + colorama.Style.RESET_ALL)
                continue

    return error_messages


def check_notebook(filename, all_sol_vars):
    """
    Compares the variables of a Jupyter notebook to a reference solution.

    :param filename: of the notebook to check.
    :param all_sol_vars: dict which contains the reference solutions.
    :return: True when all checks were successful or no checks could be performed.
    """
    print('Testing notebook ' + filename)

    # Find all solutions for the current notebook (usually more than one for TensorFlow code)
    solutions = sorted([solution for solution in all_sol_vars.keys() if solution.startswith(filename)])

    if not solutions:
        print('No matching solution for the file %s found. The notebook is skipped.\n' % filename)
        return True

    # Load the notebook file
    with open(filename, encoding='utf-8') as file:
        nb = nbformat.read(file, as_version=4)

    # Keep only the code cells (and especially remove the raw cells)
    nb.cells[:] = [cell for cell in nb.cells if cell.cell_type == 'code']

    # Convert the notebook to a Python script
    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(nb)
    source = re.sub('(.*?)get_ipython', r'#\1get_ipython', source)  # Comment out code lines which are only available in ipython
    source = re.sub('^(plotly\.offline)', r'#\1', source, flags=re.MULTILINE)  # Disable Plotly commands since they may cause problems

    # Run the student's solution
    stud_vars = {}
    with open(os.devnull, "w") as stream, contextlib.redirect_stdout(stream):
        # print commands are prevented (warnings will still be shown)
        exec(source, stud_vars)

    # Test each solution
    messages = {}
    correct_solution = ''
    for solution in solutions:
        messages[solution] = check_vars(stud_vars, all_sol_vars[solution])

        if not messages[solution]:
            correct_solution = solution

            # The first check was already successful; no need to check other possible solutions
            break

    if correct_solution:
        print(f'{colorama.Fore.GREEN}The test for the notebook %s was successful (checked against %s). No errors found.{colorama.Style.RESET_ALL}\n' % (filename, correct_solution))
        return True
    else:
        # All solutions are incorrect. Show the errors for the first solution (arbitrary)
        print('The test for the notebook %s was not successful. At least one variable does not contain the expected result (errors compared to the solution %s are shown).' % (filename, solutions[0]))

        for message in messages[solutions[0]]:
            print(message)

        return False


def version_info():
    packages = ['numpy', 'scikit-learn']
    versions = {}
    for package in packages:
        versions[package] = pkg_resources.get_distribution(package).version

    versions['python'] = platform.python_version()

    return versions


def check_cwd():
    print('Current working directory: ' + os.getcwd())

    folder_name = os.path.basename(os.getcwd())
    print('Testing all notebooks in the folder ' + folder_name)

    solution_filename = 'solution_vars.pickle'
    if not os.path.exists(solution_filename):
        print('No pickle file containing the solution data found. The folder %s is skipped.' % folder_name)
        return

    # Load the official solution data
    all_sol_vars = pickle.load(open(solution_filename, 'rb'))

    # First, check whether the same packages were used
    for package, version in version_info().items():
        if version != all_sol_vars['versions'][package]:
            print_warning('You are using %s in version %s but the solution variables were created with version %s. Results may differ.' % (package, version, all_sol_vars['versions'][package]))

    # All notebooks in the current folder
    notebook_filenames = glob.glob('*.ipynb')

    # Test each notebook where a corresponding solution is available
    checks = []
    for filename in notebook_filenames:
        successful = check_notebook(filename, all_sol_vars)
        checks.append(successful)

    # Find all available solution names
    solutions = set()
    for sol in all_sol_vars.keys():
        match = re.search(r'^[^.]+\.ipynb', sol)
        if match:
            solutions.add(match.group(0))

    # Check if the students provided a notebook for each solution
    all_sols_checked = True
    for sol in solutions:
        if sol not in notebook_filenames:
            print_warning(f'A notebook with the name {colorama.Style.NORMAL}%s{colorama.Style.BRIGHT} could not be found in the folder %s but is part of this assignment. You have either not finished your implementation yet or you used the wrong name for your notebook.' % (sol, folder_name))
            all_sols_checked = False

    if all(checks) and all_sols_checked:
        print(f'{colorama.Fore.GREEN}Congratulations! The checked notebooks in the folder %s match with the reference implementation.{colorama.Style.RESET_ALL}\n' % folder_name)


def check_folder(folder):
    prev_cwd = os.getcwd()
    os.chdir(folder)  # Notebooks should be executed in their corresponding folder
    sys.path.append(os.getcwd())  # Modules of the folder should be accessible

    check_cwd()

    os.chdir(prev_cwd)

colorama.init()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='This script tests all notebooks in one or more folders. Requirement is that the respective folder contains a file with the name solution_vars.pickle which contains the solution variables of the notebooks (a folder may contain multiple notebooks). The idea is that each folder contains all notebooks from one assignment.')
parser.add_argument('folders', type=str, nargs='*', default='.', help='Folders to check (each with its own solution file). If no folder name is given, the current folder is used.')
__version__ = '0.7.10'
parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)

args = parser.parse_args()
print(os.path.basename(__file__) + ' ' + __version__)

# Check all notebooks in the given folder name(s)
for folder in args.folders:
    check_folder(folder)
