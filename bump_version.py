from tomlkit import parse, dumps

import sys
from pathlib import Path

version = sys.argv[1]
path = Path("pyproject.toml")
doc = parse(path.read_text())

doc["project"]["version"] = version
path.write_text(dumps(doc))
