import markdown

with open("logic.txt", "r") as input_file:
    text = input_file.read()
html = markdown.markdown(text)

with open("logic.html", "w", errors="xmlcharrefreplace") as output_file:
    output_file.write(html)
