import os
import subprocess
import sys


def run_script(script_name, *args):
    subprocess.run(
        ['python', os.path.join(os.path.dirname(__file__), script_name)] + list(args)
    )  # Run the script with additional arguments


def print_help():
    print('Usage: python edgelab <task> [<args>]')
    print('Available tasks:')
    print('  train:    Run the train')
    print('  export:   Run the export')
    print('  inference: Run the inference')


def main():
    # Get command-line arguments
    if len(sys.argv) < 2:
        print('Please provide the task parameter')  # Promote user to provide task parameter
        return

    task = sys.argv[1]  # Get the task parameter
    args = sys.argv[2:]  # Get the additional arguments

    if task == 'train':
        run_script('train.py', *args)  # Run train.py script with additional arguments
    elif task == 'export':
        run_script('export.py', *args)  # Run export.py script with additional arguments
    elif task == 'inference':
        run_script('inference.py', *args)  # Run inference.py script with additional arguments
    else:
        print('Unknown task parameter')
        print_help()


if __name__ == '__main__':
    main()
