from __future__ import annotations

import ast
import hashlib
import textwrap
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class FunctionRecord:
    repo: str
    path: str
    func_name: str
    sig_text: str
    body_text: str
    full_text: str
    docstring: str
    n_body_stmts: int
    ast_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def _docstring_of(node: ast.AST) -> str:
    return ast.get_docstring(node) or ""


def _sig_from_source(source: str, node: ast.AsyncFunctionDef | ast.FunctionDef) -> str:
    """Rebuild a compact signature + docstring header from the AST node."""
    lines = source.splitlines()
    # decorator lines + def line
    start = node.lineno - 1
    # end of signature is where body starts
    if not node.body:
        return lines[start]
    body_start = node.body[0].lineno - 1
    header = "\n".join(lines[start:body_start]).rstrip()
    doc = _docstring_of(node)
    if doc and '"""' not in header and "'''" not in header:
        header = f'{header}\n    """{doc}"""'
    return header


def _body_from_source(source: str, node: ast.AsyncFunctionDef | ast.FunctionDef) -> Optional[str]:
    if not node.body:
        return None
    # Skip docstring statement if present
    body_nodes = list(node.body)
    if (
        body_nodes
        and isinstance(body_nodes[0], ast.Expr)
        and isinstance(getattr(body_nodes[0], "value", None), ast.Constant)
        and isinstance(body_nodes[0].value.value, str)
    ):
        body_nodes = body_nodes[1:]
    if not body_nodes:
        return None
    start = body_nodes[0].lineno - 1
    end = body_nodes[-1].end_lineno  # exclusive slice uses end_lineno
    lines = source.splitlines()
    body = "\n".join(lines[start:end])
    return textwrap.dedent(body).strip()


def _count_stmts(node: ast.AsyncFunctionDef | ast.FunctionDef) -> int:
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return len(body)


def extract_functions(
    source: str,
    *,
    repo: str = "",
    path: str = "",
    min_body_stmts: int = 3,
) -> list[FunctionRecord]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    out: list[FunctionRecord] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Skip nested-only; top-level functions for POC
        n_stmts = _count_stmts(node)
        if n_stmts < min_body_stmts:
            continue
        body = _body_from_source(source, node)
        if not body:
            continue
        sig = _sig_from_source(source, node)
        full = ast.get_source_segment(source, node) or f"{sig}\n{body}"
        # Normalize for near-dupe detection
        try:
            normalized = ast.dump(node, annotate_fields=False)
        except Exception:
            normalized = full
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        out.append(
            FunctionRecord(
                repo=repo,
                path=path,
                func_name=node.name,
                sig_text=sig,
                body_text=body,
                full_text=full,
                docstring=_docstring_of(node),
                n_body_stmts=n_stmts,
                ast_hash=digest,
            )
        )
    return out
