"""Compare two parameter inventories produced by model_parity.py."""

import json
import pathlib
import sys

left = json.loads(pathlib.Path(sys.argv[1]).read_text())
right = json.loads(pathlib.Path(sys.argv[2]).read_text())

lp = {name: tuple(shape) for name, shape in left["params"]}
rp = {name: tuple(shape) for name, shape in right["params"]}

print(f"class:  {left['class']} vs {right['class']}")
print(f"params: {left['total_params']} vs {right['total_params']}")
print(f"tensors:{left['n_tensors']} vs {right['n_tensors']}")

only_left = sorted(set(lp) - set(rp))
only_right = sorted(set(rp) - set(lp))
shape_diff = sorted(n for n in set(lp) & set(rp) if lp[n] != rp[n])

# The refactor renamed containers (encoder_blocks -> blocks). A rename is not an
# architecture change as long as the ordered shapes line up one to one.
shapes_left = [tuple(shape) for _, shape in left["params"]]
shapes_right = [tuple(shape) for _, shape in right["params"]]
same_layout = shapes_left == shapes_right

renames = []
if only_left and len(only_left) == len(only_right):
    for a, b in zip(sorted(only_left), sorted(only_right)):
        if a.split(".")[-3:] == b.split(".")[-3:]:
            renames.append((a, b))

print("names only in left :", len(only_left))
print("names only in right:", len(only_right))
print("shape mismatches   :", shape_diff or "none")
print("ordered shapes identical:", same_layout)
print("pure renames (same suffix, same order):", len(renames), "of", len(only_left))
if renames:
    print("  e.g.", renames[0][0], "->", renames[0][1])
print("identical initial weights (values only):", left["values_sha256"] == right["values_sha256"])

if shape_diff or not same_layout:
    print("VERDICT: architectures differ — metrics are not comparable")
    raise SystemExit(1)
if only_left and len(renames) != len(only_left):
    print("VERDICT: names differ in a way that is not a pure rename — check manually")
    raise SystemExit(1)
print("VERDICT: same architecture, same shapes, same order"
      + (f" ({len(renames)} tensors renamed)" if renames else ""))
