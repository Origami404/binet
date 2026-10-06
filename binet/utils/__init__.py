import json
import subprocess


def demangle(name: str | list[str]) -> str | list[str]:
    """Demangle one name or a batch, preserving input order and unavailable names."""
    if isinstance(name, str):
        names = [name]
    elif isinstance(name, list) and all(isinstance(item, str) for item in name):
        names = name
    else:
        raise TypeError("name must be a string or a list of strings")
    # stdin uses one symbol per line; preserve names that cannot use that framing.
    mangled = list(dict.fromkeys(item for item in names
                                if item.startswith("_Z") and "\n" not in item and "\r" not in item))
    decoded = {}
    if mangled:
        try:
            result = subprocess.run(["c++filt", "-n"], input="\n".join(mangled) + "\n",
                                    capture_output=True, text=True, timeout=10)
            lines = [line.strip() for line in result.stdout.splitlines()]
            if result.returncode == 0 and len(lines) == len(mangled) and all(lines):
                decoded = dict(zip(mangled, lines))
        except (OSError, subprocess.TimeoutExpired, UnicodeError):
            pass
    output = [decoded.get(item, item) for item in names]
    return output[0] if isinstance(name, str) else output


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
