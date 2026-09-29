"""Tables and charts in the terminal.  ASCII only, so it prints the same on Windows, Mac and Linux."""
import os
import textwrap

import pandas as pd

from src.config import OUTPUTS_DIR, ROOT

WIDTH = 110
DEMO_DIR = OUTPUTS_DIR / "demo"


def banner(text, ch="="):
    print("\n" + ch * WIDTH + f"\n{text}\n" + ch * WIDTH)


def step(n, title, what):
    banner(f"STEP {n}: {title}")
    for line in textwrap.wrap(what, WIDTH - 2):
        print(" " + line)
    print()


def say(text, indent=1):
    for line in textwrap.wrap(str(text), WIDTH - 2 * indent):
        print(" " * indent + line)


def bullets(items, indent=2):
    for it in items:
        lines = textwrap.wrap(str(it), WIDTH - indent - 2) or [""]
        print(" " * indent + "- " + lines[0])
        for l in lines[1:]:
            print(" " * (indent + 2) + l)


def bar(value, width=30, mark=None):
    n = int(round(max(0, min(1, value)) * width))
    b = "#" * n + "." * (width - n)
    if mark is not None:
        i = int(round(mark * width))
        b = b[:i] + "|" + b[i + 1:] if 0 <= i < width else b
    return "[" + b + f"] {value:.2f}"


def table(df: pd.DataFrame, title=None, max_rows=15, digits=3, max_col=34):
    if title:
        print(f" {title}")
    if df is None or len(df) == 0:
        print("   (empty)\n")
        return
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
        else:
            d[c] = d[c].map(lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        d[c] = d[c].map(lambda s: s if len(s) <= max_col else s[:max_col - 2] + "..")
    w = {c: max(len(str(c)), d[c].str.len().max()) for c in d.columns}
    line = "+" + "+".join("-" * (w[c] + 2) for c in d.columns) + "+"
    print(line)
    print("| " + " | ".join(str(c).ljust(w[c]) for c in d.columns) + " |")
    print(line)
    for _, r in d.iterrows():
        print("| " + " | ".join(r[c].ljust(w[c]) for c in d.columns) + " |")
    print(line)
    if len(df) > max_rows:
        print(f"  ... {len(df) - max_rows} more row(s)")
    print()


def show(fig, name, window=True):
    """Save the chart as a PNG and, if a window is wanted, open it (close the window to continue)."""
    import matplotlib.pyplot as plt
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    p = DEMO_DIR / f"{name}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    try:
        shown = p.relative_to(ROOT)
    except ValueError:
        shown = p
    print(f"   [chart saved] {shown}")
    if window and os.environ.get("FRAUDDEX_SHOW") == "1":
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


def pause(enabled):
    if enabled:
        input("\n   Press Enter to continue... ")


def print_case(res):
    banner(f"{res['case_id']}   {res['merged'].get('Patient_Name', '')}   {res['merged'].get('Patient_ID', '')}", "-")
    print(f" VERDICT : {res['verdict']}")
    print(f" RISK    : {bar(res['risk'], mark=res['threshold'])}   ('|' marks the review threshold {res['threshold']:.2f})")
    print(f" from    : model {res['ml_probability']:.2f}   rules {res['rule_score']:.2f}" + (f"   expected cost {res['expected_cost']:,.0f}" if res.get("expected_cost") else ""))
    ex = res.get("explanation", {})
    if ex:
        say(ex["headline"])
    print()
    ab = pd.DataFrame([{"severity": a["severity"], "from": a["source"], "what": a["title"], "detail": a["detail"]} for a in ex.get("abnormal", [])])
    table(ab, f"WHAT LOOKS ABNORMAL ({len(ab)})", max_rows=10, max_col=52)
    nm = pd.DataFrame(ex.get("normal", []))
    table(nm, f"WHAT LOOKS NORMAL ({len(nm)})", max_col=60)
    if res["documents"]:
        dt = pd.DataFrame([{"file": d["file"], "type": d["doc_type"], "abnormality": d["abnormality"], "findings": "; ".join(f["code"] for f in d["findings"]) or "none"} for d in res["documents"]])
        table(dt, "DOCUMENTS", max_col=40)
    if res["crosscheck"]["items"]:
        cc = pd.DataFrame(res["crosscheck"]["items"])[["field", "entered", "document", "status"]]
        table(cc, "ENTERED vs DOCUMENT", max_col=24)
    h = res["history"]
    if h.get("patient_cases_total", 0) or h.get("provider_claims_total", 0):
        print(f" HISTORY : patient has {h['patient_cases_total']:.0f} earlier claim(s) ({h['patient_claims_90d']:.0f} in 90 days); "
              f"provider has {h['provider_claims_total']:.0f} earlier claim(s) ({h['provider_claims_30d']:.0f} in 30 days).\n")
    if ex.get("checks"):
        print(" WHAT TO CHECK (retrieved from the knowledge base):")
        bullets(ex["checks"][:5])
        print()