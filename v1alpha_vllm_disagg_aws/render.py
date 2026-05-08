import yaml
import jinja2
from pathlib import Path

TEMPLATE = Path("template.yaml")
VALUES   = Path("values.yaml")
OUTPUT   = Path("rendered.yaml")

with VALUES.open() as f:
    values = yaml.safe_load(f)

with TEMPLATE.open() as f:
    rendered = jinja2.Template(f.read()).render(**values)

with OUTPUT.open("w") as f:
    f.write(rendered)

print(f"Rendered -> {OUTPUT}")
