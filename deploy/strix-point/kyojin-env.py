#!/usr/bin/env python3
"""Print Kyojin's measured Qwen serving env (SERVE_ENV in tools/qwen/serve.py) as `export K=V` lines, without
importing the server. Variables already set in the caller's environment are left out, matching its setdefault.
usage: eval "$(kyojin-env.py TREE)" """
import ast
import os
import shlex
import sys

src = open(os.path.join(sys.argv[1], "tools/qwen/serve.py")).read()
for node in ast.parse(src).body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SERVE_ENV" for t in node.targets):
        for k, v in ast.literal_eval(node.value):
            if k not in os.environ:
                print(f"export {k}={shlex.quote(v)}")
        break
else:
    sys.exit("SERVE_ENV not found")
