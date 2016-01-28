import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file_name", help="The file name")
parser.add_argument("optional_arguments", help="The option arguments (hidden nodes and holdhout percent)", nargs='*')
args = parser.parse_args()
print(args.file_name)
if(args.optional_arguments):
    print (args.optional_arguments)

