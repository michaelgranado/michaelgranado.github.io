import argparse

HTML_HEADER = """<!DOCTYPE html>
<html>
    <head>
        <link rel="stylesheet" href="/styles.css">
    </head>
    <body>
        <pre>
"""

HTML_FOOTER = """       </pre>
    </body>
</html>
"""

parser = argparse.ArgumentParser(
                    prog='HTMLFormatter',
                    description='Formats Files into HTML to Display on WebPage',
                    epilog='Takes a commandline arg for the file to convert')
parser.add_argument('filename') 
args = parser.parse_args()

split_filename = args.filename.split(".")

if len(split_filename) == 1:
    print("Error: No File Extension Provided")
    exit(1)

new_filename = split_filename[0] + '.html'
try:
    with open(args.filename, 'r') as old_file, open(new_filename, 'w') as new_file:
        new_file.write(HTML_HEADER) 
        
        # Read line by line
        for line in old_file:
            new_file.write(line)

        new_file.write(HTML_FOOTER)
except FileNotFoundError:
    print("Error: File not found.")
