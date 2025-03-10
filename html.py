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
parser.add_argument('out_name')
args = parser.parse_args()

split_filename = args.filename.split(".")

if len(split_filename) == 1:
    print("Error: No File Extension Provided")
    exit(1)

file_type = split_filename[0]

try:
    with open(args.filename, 'r') as old_file, open(args.out_name, 'w') as new_file:
        if file_type == "logic":
            new_file.write('<body style="background: lightyellow;">\n')
        else:
            new_file.write(HTML_HEADER)
        
        marker = False
        # Read line by line
        for line in old_file:
            if file_type == "logic" and '```' in line:
                if marker == False:
                    line = "<pre>\n"
                    marker = True
                else:
                    line = "</pre>\n"
                    marker = False
            if (file_type != "logic" or (file_type == "logic" and marker)) and ("<" in line or ">" in line):
                line.replace("<", "&lt;")
                line.replace(">", "&gt;")

            new_file.write(line)
        
        if file_type == "logic":
            new_file.write('</body>\n')
        else:
            new_file.write(HTML_FOOTER)
except FileNotFoundError:
    print("Error: File not found.")
