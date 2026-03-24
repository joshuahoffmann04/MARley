"""Generate a complete UML class diagram for the MARley package.

Uses pyreverse (pylint) to analyze the code, then transforms the output
so that classes are grouped inside their package namespaces.

Usage:
    python docs/uml/generate_uml.py          # generates .puml + .png
    python docs/uml/generate_uml.py --puml   # generates .puml only (no Java needed)

Requirements:
    - pylint  (pip install pylint)
    - Java + plantuml.jar for PNG rendering (auto-downloaded if missing)
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # MARley/
SRC = ROOT / "src"
OUT_DIR = Path(__file__).resolve().parent  # docs/uml/
PLANTUML_JAR = OUT_DIR / "plantuml.jar"
PLANTUML_URL = (
    "https://github.com/plantuml/plantuml/releases/download/v1.2024.8/"
    "plantuml-1.2024.8.jar"
)

# Package display colors (Material Design palette)
PKG_COLORS = {
    "models":     "#FFF8E1",
    "extractor":  "#E8F5E9",
    "chunker":    "#E3F2FD",
    "retrieval":  "#FFF3E0",
    "generator":  "#F3E5F5",
    "abstention": "#FFEBEE",
    "server":     "#E0F7FA",
}

# Class colors per package
CLS_COLORS = {
    "models":     "#FFE082",
    "extractor":  "#A5D6A7",
    "chunker":    "#90CAF9",
    "retrieval":  "#FFCC80",
    "generator":  "#CE93D8",
    "abstention": "#EF9A9A",
    "server":     "#80DEEA",
}


def run_pyreverse() -> str:
    """Run pyreverse and return the classes .puml content."""
    # Temporarily create src/__init__.py so pyreverse resolves src.marley.* imports
    init_file = SRC / "__init__.py"
    created = False
    if not init_file.exists():
        init_file.write_text("")
        created = True
    try:
        subprocess.run(
            ["pyreverse", "-o", "puml", "-p", "MARley", "--colorized", str(SRC)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        if created:
            init_file.unlink(missing_ok=True)

    puml_path = ROOT / "classes_MARley.puml"
    content = puml_path.read_text(encoding="utf-8")
    # Clean up temp files
    puml_path.unlink(missing_ok=True)
    (ROOT / "packages_MARley.puml").unlink(missing_ok=True)
    return content


def parse_classes(puml: str) -> tuple[list[dict], list[str]]:
    """Parse pyreverse puml into class definitions and relationship lines."""
    classes: list[dict] = []
    relations: list[str] = []

    # Match class blocks
    class_pattern = re.compile(
        r'class\s+"([^"]+)"\s+as\s+(\S+)\s+#\w+\s*\{([^}]*)\}',
        re.DOTALL,
    )
    for m in class_pattern.finditer(puml):
        display_name = m.group(1)
        qualified = m.group(2)
        body = m.group(3)
        classes.append({
            "display_name": display_name,
            "qualified": qualified,
            "body": body,
        })

    # Match relationship lines (after class definitions)
    for line in puml.splitlines():
        line = line.strip()
        if re.match(r"^src\.marley\.\S+\s+--", line):
            relations.append(line)

    return classes, relations


def get_package(qualified: str) -> str:
    """Extract the marley sub-package from a qualified name.

    src.marley.models.retrieval.Retriever -> models
    src.marley.server.models.ChatRequest -> server
    """
    parts = qualified.split(".")
    # src.marley.<package>.…
    if len(parts) >= 3:
        return parts[2]  # e.g. 'models', 'retrieval', etc.
    return "root"


def build_puml(classes: list[dict], relations: list[str]) -> str:
    """Build a PlantUML string with classes grouped inside packages."""
    # Group classes by package
    packages: dict[str, list[dict]] = {}
    for cls in classes:
        pkg = get_package(cls["qualified"])
        packages.setdefault(pkg, []).append(cls)

    lines = [
        "@startuml marley_uml",
        "",
        "skinparam classAttributeIconSize 0",
        "skinparam packageStyle rectangle",
        "skinparam defaultFontSize 11",
        "skinparam classFontSize 12",
        "skinparam packageFontSize 14",
        "skinparam packageFontStyle bold",
        "skinparam shadowing false",
        "skinparam class {",
        "  BorderColor #666666",
        "  ArrowColor #444444",
        "}",
        "",
        'title MARley -- UML Class Diagram',
        "",
    ]

    # Emit packages with their classes
    pkg_order = ["models", "extractor", "chunker", "retrieval", "generator", "abstention", "server"]
    for pkg_name in pkg_order:
        if pkg_name not in packages:
            continue
        pkg_color = PKG_COLORS.get(pkg_name, "#EEEEEE")
        cls_color = CLS_COLORS.get(pkg_name, "#99DDFF")
        lines.append(f'package "marley.{pkg_name}" {pkg_color} {{')

        for cls in packages[pkg_name]:
            alias = cls["qualified"].replace("src.marley.", "marley.")
            body = cls["body"]
            lines.append(f'  class "{cls["display_name"]}" as {alias} {cls_color} {{{body}}}')

        lines.append("}")
        lines.append("")

    # Emit relationships with cleaned aliases
    lines.append("' ── Relationships ──")
    for rel in relations:
        rel = rel.replace("src.marley.", "marley.")
        lines.append(rel)

    lines.append("")
    lines.append("@enduml")
    return "\n".join(lines)


def render_png(puml_path: Path) -> Path:
    """Render .puml to .png using PlantUML."""
    if not PLANTUML_JAR.exists():
        print(f"Downloading PlantUML jar...")
        urllib.request.urlretrieve(PLANTUML_URL, str(PLANTUML_JAR))

    png_path = puml_path.with_suffix(".png")
    subprocess.run(
        ["java", "-jar", str(PLANTUML_JAR), "-tpng", str(puml_path)],
        cwd=str(OUT_DIR),
        check=True,
    )
    return png_path


def main():
    puml_only = "--puml" in sys.argv

    print("Running pyreverse...")
    raw_puml = run_pyreverse()

    print("Parsing classes and relationships...")
    classes, relations = parse_classes(raw_puml)
    print(f"  Found {len(classes)} classes, {len(relations)} relationships")

    print("Building grouped PlantUML...")
    puml_content = build_puml(classes, relations)

    puml_path = OUT_DIR / "marley_uml.puml"
    puml_path.write_text(puml_content, encoding="utf-8")
    print(f"  Written: {puml_path}")

    if puml_only:
        print("Done (--puml mode, skipping PNG).")
        return

    print("Rendering PNG...")
    png_path = render_png(puml_path)
    print(f"  Written: {png_path}")
    print("Done.")


if __name__ == "__main__":
    main()
