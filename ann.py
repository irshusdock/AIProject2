import argparse

"Parse through the list of arguements and return the value of the passed argument"
"list_of_args is the list of arguments in the format [arg1, val1, arg2, val2, etc]"
"arg is the argument to search for (e.x. arg1, arg2 in above)"
def get_argument(list_of_args, arg):
    if(not list_of_args):
        return 0
    counter = 0
    for element in list_of_args:
        if element == arg:
            return list_of_args[counter+1]
        counter = counter + 1
    return 0


parser = argparse.ArgumentParser()
parser.add_argument("file_name", help="The file name")
parser.add_argument("optional_arguments", help="The option arguments (hidden nodes and holdhout percent)", nargs='*')
args = parser.parse_args()

"Parse the arguments"
file_name = args.file_name
hidden_nodes = get_argument(args.optional_arguments, 'h')
holdout_percent = get_argument(args.optional_arguments, 'p')

