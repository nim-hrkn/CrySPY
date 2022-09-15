with open("../read_input.py") as handle:
    content = handle.read().splitlines()

import re
variables = []
for line in content:
    if re.match(f'^ *global ',line):
        line = line.replace('global','')
        for name in line.split(","):
            name = name.strip()
            variables.append(name)


with open('var_def.py','w') as handle:
    lines = []
    for name in variables:
        lines.append(f'self.{name} = {name}')
    handle.write("\n".join(lines)+"\n")

with open('use_def.py','w') as handle:
    lines = []
    for name in variables:
        lines.append(f'{name} = self.{name}')
    handle.write("\n".join(lines)+"\n")

with open('def_none.py','w') as handle:
    lines = []
    for name in variables:
        lines.append(f'{name} = None')
    handle.write("\n".join(lines)+"\n")