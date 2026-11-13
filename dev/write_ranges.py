"""Write ranges (JSON from dev/report_ranges.py or dev/crosscheck_ranges.py) into a module's
REFERENCE dict -- only inside that dict, never elsewhere in the file.

usage: write_ranges.py solution/report.py ranges.json
"""
import json, re, sys

path, js = sys.argv[1], sys.argv[2]
new = json.load(open(js))
s = open(path).read()
a = s.index("REFERENCE = {")
b = s.index("\n}\n", a) + 3
block = s[a:b]
n = 0
for name, (lo, hi) in new.items():
    pat = re.compile(r'(    "%s": )\([0-9.]+, [0-9.]+\)' % re.escape(name))
    block, k = pat.subn(lambda m: m.group(1) + "(%s, %s)" % (lo, hi), block)
    n += k
open(path, "w").write(s[:a] + block + s[b:])
print(path, "ranges written:", n)
