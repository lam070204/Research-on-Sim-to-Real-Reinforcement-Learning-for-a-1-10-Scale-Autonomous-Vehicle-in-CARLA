import io
import os
import sys
import tokenize

file_list = sys.argv[1]
ok = 0
bad = 0

with io.open(file_list, "r", encoding="utf-8-sig") as f:
    paths = [line.strip() for line in f if line.strip()]

for path in paths:
    try:
        with tokenize.open(path) as src:
            text = src.read()
        compile(text, path, "exec")
        print("[OK]   " + path)
        ok += 1
    except Exception as exc:
        print("[FAIL] " + path)
        print("       " + repr(exc))
        bad += 1

print("")
print("SUMMARY: OK={} FAIL={} TOTAL={}".format(ok, bad, ok + bad))
sys.exit(1 if bad else 0)
