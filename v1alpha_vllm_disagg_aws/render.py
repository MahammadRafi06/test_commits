import yaml
import jinja2
from pathlib import Path

TEMPLATE = Path("template.yaml")
VALUES   = Path("values.yaml")
OUTPUT   = Path("rendered.yaml")

DEL = "DEL"

def remove_del(obj):
    if isinstance(obj, dict):
        return {k: remove_del(v) for k, v in obj.items() if v != DEL}
    if isinstance(obj, list):
        return [remove_del(i) for i in obj if i != DEL]
    return obj

with VALUES.open() as f:
    values = yaml.safe_load(f)

with TEMPLATE.open() as f:
    rendered = jinja2.Template(f.read()).render(**values)

doc = remove_del(yaml.safe_load(rendered))

with OUTPUT.open("w") as f:
    yaml.dump(doc, f, default_flow_style=False, allow_unicode=True)

print(f"Rendered -> {OUTPUT}")
