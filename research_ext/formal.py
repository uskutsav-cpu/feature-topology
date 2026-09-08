"""Explicit Lean build status, graph export and dependency auditing."""
from __future__ import annotations
from pathlib import Path
import re
import shutil
import subprocess
from .io import atomic_text, atomic_json, digest, file_digest, utc_now
from .exact import verify_polygon_certificate


def export_graph(cert: dict, path: str | Path) -> dict:
    if not verify_polygon_certificate(cert):
        raise ValueError("Refusing to export an invalid exact certificate")
    graph = cert["graph"]
    if graph["vertex_count"] > 64:
        raise ValueError("Kernel-reduced graph examples are limited to 64 vertices")
    edges = ", ".join(f"({a}, {b})" for a,b in graph["edges"])
    code = f'''import FeatureTopology.Graph

/- Certificate SHA256: {digest(cert)}.
   This checks the abstract supplied graph, not Python's geometric construction. -/
namespace FeatureTopology.Generated

def certificateGraph : FeatureTopology.FiniteGraph :=
  ⟨{graph['vertex_count']}, [{edges}]⟩

theorem certificate_graph_checked :
    FeatureTopology.checkGraph certificateGraph {graph['beta0']} {graph['beta1']} = true := by decide

end FeatureTopology.Generated
'''
    atomic_text(path, code)
    return {"source": str(path), "source_sha256": file_digest(path),
            "status": "lean_source_generated_not_compiled", "certificate_sha256": digest(cert)}


def _without_comments(text: str) -> str:
    """Handle nested Lean block comments; retain tokens outside comments."""
    output, depth, index = [], 0, 0
    while index < len(text):
        if text.startswith('/-', index):
            depth += 1; index += 2
        elif depth and text.startswith('-/', index):
            depth -= 1; index += 2
        elif depth:
            index += 1
        elif text.startswith('--', index):
            end = text.find('\n', index)
            index = len(text) if end < 0 else end
        else:
            output.append(text[index]); index += 1
    if depth:
        raise ValueError("Unterminated Lean comment")
    return ''.join(output)


def check_formal(root: str | Path, output: str | Path, *, timeout: int = 180) -> dict:
    root, output = Path(root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = {str(p.relative_to(root)): file_digest(p) for p in root.rglob('*.lean')
             if '.lake' not in p.parts}
    forbidden = []
    for relative in files:
        source = _without_comments((root/relative).read_text())
        if re.search(r'\b(sorry|admit|axiom|native_decide|unsafe)\b', source):
            forbidden.append(relative)
    result = {"utc": utc_now(), "files": files, "lean_verified": False,
              "scope": "Logical lemmas and the finite supplied-graph checker only."}
    required = {'FeatureTopology.lean','FeatureTopology/Audit.lean','FeatureTopology/Generated.lean'}
    missing = sorted(required - files.keys())
    if forbidden:
        result.update(status='rejected_forbidden_constructs', files_with_forbidden_constructs=forbidden)
    elif missing or not (root/'lean-toolchain').is_file() or not (root/'lakefile.toml').is_file():
        result.update(status='incomplete_formal_project', missing_sources=missing)
    elif shutil.which('lake') is None:
        result.update(status='not_run_no_lean_toolchain', detail='No lake executable in this runtime')
    else:
        logs = []
        try:
            for command in [['lake','build'], ['lake','env','lean','FeatureTopology/Audit.lean']]:
                process = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=timeout)
                logs.append('$ '+' '.join(command)+'\n'+process.stdout+process.stderr)
                if process.returncode:
                    raise RuntimeError(f"Command failed with status {process.returncode}: {command}")
            log='\n'.join(logs)
            if any(word in log for word in ['sorryAx', 'Lean.ofReduceBool', 'Lean.trustCompiler', '_native.native_decide']):
                raise RuntimeError('Forbidden proof dependency in Lean output')
            expected = re.findall(r'#print\s+axioms\s+([A-Za-z0-9_.]+)',
                                  (root/'FeatureTopology/Audit.lean').read_text())
            if len(expected) < 14 or 'FeatureTopology.Generated.certificate_graph_checked' not in expected:
                raise RuntimeError('The required theorem audit list is incomplete')
            for name in expected:
                if not re.search(r"'"+re.escape(name)+r"' (depends on axioms:|does not depend on any axioms)", log):
                    raise RuntimeError(f'Missing dependency audit output for {name}')
            # Accept only the three standard logical axioms. No theorem-specific axioms.
            for match in re.findall(r'depends on axioms:\s*\[([^]]*)\]', log):
                names={n.strip() for n in match.split(',') if n.strip()}
                if names-{'propext','Classical.choice','Quot.sound'}:
                    raise RuntimeError(f'Unexpected axioms: {names}')
            result.update(status='compiled_and_axiom_audited', lean_verified=True)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            result.update(status='verification_failed', detail=str(exc))
        atomic_text(output/'lean_build.log','\n'.join(logs))
    atomic_json(output/'formal_status.json',result)
    return result
