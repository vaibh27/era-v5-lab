"""Registry viewer: every rule and its function on one page.

Usage:
    python -I registry_view.py [--registry <registry.json>] [--out <registry.html>]
    python -I registry_view.py --show DA-B3-001

Writes one HTML file (default: registry.html next to registry.json). Open it in a browser: no server and no
internet are needed. It has a table of all rules with a search box and filters (team, pack, code, action, status).
Click a row to see the rule's function. The source of the helper functions, constants and imports that the
function uses is shown under it, so the rule can be read on its own.

--show prints the same source for one rule ID in the terminal.

The rules files are only read as text and parsed with ast. Nothing in them is run. The code is shown as plain
text on the page, never as HTML.
"""
import argparse
import ast
import json
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REGISTRY = HERE.parent / "rules" / "registry.json"
GENERATED = {"RULES", "RULES_BY_ID"}  # names the extractor adds: never shown as helpers


# --- reading the source of a rule --------------------------------------------------------------------------------

def _top_level(body):
    """The top-level statements of a module. The extractor puts imports and constants inside try: blocks."""
    for node in body:
        if isinstance(node, ast.Try):
            yield from _top_level(node.body)
        else:
            yield node


def _segment(src: str, node) -> str:
    """The source text of a node, dedented (a node inside a try: block keeps its indent after the first line)."""
    return textwrap.dedent(" " * node.col_offset + (ast.get_source_segment(src, node) or "")).strip("\n")


def _dict_assigns(tree, name):
    """The dicts assigned to `name` at the top level, in file order: [{key: value node}]."""
    found = []
    for node in _top_level(tree.body):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            found.append({k.value: v for k, v in zip(node.value.keys, node.value.values)
                          if isinstance(k, ast.Constant) and isinstance(k.value, str)})
    return found


def read_rule(rules_file: Path, rule_id: str) -> dict:
    """Return {function, source, helpers} for a rule ID, or {error} if it cannot be read. Never runs the code."""
    try:
        src = rules_file.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (OSError, UnicodeDecodeError, SyntaxError) as e:
        return {"error": f"cannot read {rules_file.name}: {type(e).__name__}"}

    defs, imports = {}, {}
    for node in _top_level(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs[node.name] = node
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    defs[t.id] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                imports[(a.asname or a.name).split(".")[0]] = node

    by_id = (_dict_assigns(tree, "RULES_BY_ID") or [{}])[-1]
    value = by_id.get(rule_id)
    if value is None:
        return {"error": f"{rule_id} is not in RULES_BY_ID of {rules_file.name}"}
    if isinstance(value, ast.Name) and value.id in defs:
        main = defs[value.id]
        name = value.id
    else:  # a lambda, or anything else that is written inline
        main, name = value, "<inline>"

    # The helpers: module-level names that the rule uses, and the ones those use, in file order.
    wanted, queue = {}, [main]
    while queue:
        node = queue.pop()
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in GENERATED:
                if n.id in defs and defs[n.id] is not main and n.id not in wanted:
                    wanted[n.id] = defs[n.id]
                    queue.append(defs[n.id])
                elif n.id in imports and n.id not in wanted:
                    wanted[n.id] = imports[n.id]
    nodes = sorted({id(n): n for n in wanted.values()}.values(), key=lambda n: n.lineno)
    return {"function": name, "source": _segment(src, main), "helpers": [_segment(src, n) for n in nodes]}


# --- the page ----------------------------------------------------------------------------------------------------

def load(registry: Path) -> list:
    try:
        entries = json.loads(registry.read_text(encoding="utf-8"))["rules"]
    except (OSError, ValueError, KeyError) as e:
        raise SystemExit(f"error: cannot read the registry {registry}: {e}")
    rows = []
    for e in sorted(entries, key=lambda e: e["id"]):
        rule = read_rule(registry.parent / e["file"], e["id"])
        rows.append({"id": e["id"], "team": e["id"].split("-")[0], "pack": e["pack"], "handle": e["handle"],
                     "code": e["code"], "n": e.get("n", 1), "action": e.get("action") or "?",
                     "withdrawn": bool(e.get("withdrawn")), "file": e["file"], **rule})
    return rows


def render(rows: list) -> str:
    # In a <script> tag the data must not contain "<", ">" or "&": the code of the annotators is untrusted text.
    data = json.dumps(rows).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return PAGE.replace("__DATA__", data)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rule registry</title>
<style>
:root { --bg:#ffffff; --fg:#1b1f24; --muted:#59636e; --line:#d0d7de; --panel:#f6f8fa; --accent:#0b5fcc; --warn:#9a6700; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1318; --fg:#e6edf3; --muted:#9aa4af; --line:#2d333b; --panel:#161b22; --accent:#6cb0ff; --warn:#e3b341; }
}
* { box-sizing: border-box; }
body { margin:0; padding:16px; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; }
h1 { margin:0 0 4px; font-size:1.3rem; }
#summary { margin:0 0 12px; color:var(--muted); }
.controls { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }
.controls input, .controls select, .controls button { font:inherit; padding:6px 8px; color:var(--fg); background:var(--panel);
  border:1px solid var(--line); border-radius:6px; }
.controls input { flex:1 1 260px; min-width:0; }
.wrap { overflow-x:auto; border:1px solid var(--line); border-radius:8px; }
table { border-collapse:collapse; width:100%; }
th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
th { background:var(--panel); position:sticky; top:0; }
tr.rule { cursor:pointer; }
tr.rule:hover, tr.rule:focus { background:var(--panel); outline:none; }
tr.rule td:first-child { color:var(--accent); font-weight:600; }
tr.withdrawn td { color:var(--muted); text-decoration:line-through; }
tr.detail td { white-space:normal; background:var(--panel); }
pre { margin:6px 0 10px; padding:10px; overflow-x:auto; background:var(--bg); border:1px solid var(--line); border-radius:6px;
  font:13px/1.4 ui-monospace, Menlo, Consolas, monospace; white-space:pre; }
.label { color:var(--muted); font-size:.85rem; }
.err { color:var(--warn); }
#none { padding:16px; color:var(--muted); display:none; }
</style>
</head>
<body>
<h1>Rule registry</h1>
<p id="summary"></p>
<div class="controls">
  <input id="q" type="search" placeholder="Search id, code, annotator, function or code text" aria-label="Search">
  <select id="team" aria-label="Team"></select>
  <select id="pack" aria-label="Pack"></select>
  <select id="code" aria-label="Code"></select>
  <select id="action" aria-label="Action"></select>
  <select id="status" aria-label="Status"></select>
  <button id="toggle" type="button">Expand all</button>
</div>
<div class="wrap">
<table>
  <thead><tr><th>ID</th><th>Code</th><th>Pack</th><th>Annotator</th><th>Action</th><th>Function</th><th>Status</th></tr></thead>
  <tbody id="rows"></tbody>
</table>
<div id="none">No rule matches.</div>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const rules = JSON.parse(document.getElementById("data").textContent);
const $ = id => document.getElementById(id);
const status = r => r.withdrawn ? "withdrawn" : "active";
rules.forEach(r => {
  r.text = [r.id, r.code, r.pack, r.handle, r.function || "", r.source || "", (r.helpers || []).join("\\n")].join("\\n").toLowerCase();
});

const filters = {team: r => r.team, pack: r => r.pack, code: r => r.code, action: r => r.action, status};
for (const [id, get] of Object.entries(filters)) {
  const sel = $(id);
  const values = [...new Set(rules.map(get))].sort();
  for (const v of ["", ...values]) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v === "" ? "All " + id + "s" : v;
    sel.appendChild(o);
  }
  sel.addEventListener("change", render);
}
$("q").addEventListener("input", render);

let allOpen = false;
$("toggle").addEventListener("click", () => {
  allOpen = !allOpen;
  $("toggle").textContent = allOpen ? "Collapse all" : "Expand all";
  document.querySelectorAll("tr.detail").forEach(d => d.hidden = !allOpen);
});

function block(label, text) {
  const wrap = document.createElement("div");
  const l = document.createElement("div");
  l.className = "label";
  l.textContent = label;
  const pre = document.createElement("pre");
  pre.textContent = text;           // text, never HTML: the code is written by the annotators
  wrap.append(l, pre);
  return wrap;
}

function detail(r) {
  const tr = document.createElement("tr");
  tr.className = "detail";
  tr.hidden = !allOpen;
  const td = document.createElement("td");
  td.colSpan = 7;
  const where = document.createElement("div");
  where.className = "label";
  where.textContent = "Medicine " + r.n + " for " + r.code + " in " + r.file;
  td.appendChild(where);
  if (r.error) {
    const e = document.createElement("div");
    e.className = "err";
    e.textContent = r.error;
    td.appendChild(e);
  } else {
    td.appendChild(block("Rule: " + r.function, r.source));
    if (r.helpers.length) td.appendChild(block("Used by the rule", r.helpers.join("\\n\\n")));
  }
  tr.appendChild(td);
  return tr;
}

function render() {
  const q = $("q").value.trim().toLowerCase();
  const tbody = $("rows");
  tbody.textContent = "";
  let shown = 0;
  for (const r of rules) {
    if (q && !r.text.includes(q)) continue;
    if ((v => v && v !== r.team)($("team").value) || (v => v && v !== r.pack)($("pack").value) ||
        (v => v && v !== r.code)($("code").value) || (v => v && v !== r.action)($("action").value) ||
        (v => v && v !== status(r))($("status").value)) continue;
    shown++;
    const tr = document.createElement("tr");
    tr.className = "rule" + (r.withdrawn ? " withdrawn" : "");
    tr.tabIndex = 0;
    for (const v of [r.id, r.code, r.pack, r.handle, r.action, r.function || "-", status(r)]) {
      const td = document.createElement("td");
      td.textContent = v;
      tr.appendChild(td);
    }
    const d = detail(r);
    const flip = () => { d.hidden = !d.hidden; };
    tr.addEventListener("click", flip);
    tr.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); flip(); } });
    tbody.append(tr, d);
  }
  $("none").style.display = shown ? "none" : "block";
  const withdrawn = rules.filter(r => r.withdrawn).length;
  $("summary").textContent = rules.length + " rules (" + withdrawn + " withdrawn) from " +
    new Set(rules.map(r => r.pack)).size + " packs. Showing " + shown + ". Click a row to see the function.";
}
render();
</script>
</body>
</html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Show every rule of the registry and its function on one page.")
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--out", type=Path, help="the HTML file (default: registry.html next to the registry)")
    ap.add_argument("--show", metavar="RULE_ID", help="print the source of one rule in the terminal instead")
    args = ap.parse_args(argv)

    rows = load(args.registry)
    if args.show:
        match = [r for r in rows if r["id"] == args.show.strip().upper()]
        if not match:
            print(f"error: {args.show} is not in {args.registry}", file=sys.stderr)
            return 2
        r = match[0]
        print(f"{r['id']}  code {r['code']}  {r['pack']}/{r['handle']}  action {r['action']}"
              + ("  (withdrawn)" if r["withdrawn"] else ""))
        if "error" in r:
            print(f"  {r['error']}")
            return 1
        print(f"\n# {r['function']}\n{r['source']}")
        if r["helpers"]:
            print("\n# used by the rule\n" + "\n\n".join(r["helpers"]))
        return 0

    out = args.out or args.registry.parent / "registry.html"
    with open(out, "w", encoding="utf-8", newline="\n") as f:  # LF on every platform, as medicine_extract.py writes
        f.write(render(rows))
    broken = [r["id"] for r in rows if "error" in r]
    print(f"{len(rows)} rules ({sum(r['withdrawn'] for r in rows)} withdrawn) from {len({r['pack'] for r in rows})} packs")
    if broken:
        print(f"source not readable for: {', '.join(broken)}", file=sys.stderr)
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
