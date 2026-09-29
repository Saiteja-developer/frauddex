"""FraudDex command line.  Plain `python` only; no pip or streamlit command needed on PATH.

    python run.py check                 what is installed
    python run.py scan                  what is inside data/ and what can be trained
    python run.py train [--quick]       train on everything found in data/
    python run.py demo                  show how the models and rules work, with tables and charts
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def cmd_check(a):
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_env
    check_env.main()


def cmd_scan(a):
    from src.config import ensure_dirs
    from src.training.datasets import scan_data
    ensure_dirs()
    found = scan_data()
    if not found:
        print("data/ is empty. Add your training files, then run:  python run.py train")
    for s in found:
        print(f"  {s.kind:10s} {s.name:26s} {s.note}" + (f"   label column: {s.label}" if s.label and s.kind in ("claim", "custom") else ""))


def cmd_train(a):
    from src.config import ensure_dirs
    from src.training.datasets import train_all
    ensure_dirs()
    done = train_all(quick=a.quick)
    print("\nTrained:", ", ".join(s.name for s in done) or "nothing")
    print("Reports: models/registry/<name>/report.html      Next: python run.py demo")


def cmd_demo(a):
    if not a.no_window:
        os.environ["FRAUDDEX_SHOW"] = "1"
    from src.reporting import demo
    demo.run(window=not a.no_window, pause=a.pause, only=a.only)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(f=cmd_check)
    sub.add_parser("scan").set_defaults(f=cmd_scan)
    t = sub.add_parser("train"); t.add_argument("--quick", action="store_true", help="fewer trees, about 1 minute"); t.set_defaults(f=cmd_train)
    d = sub.add_parser("demo")
    d.add_argument("--no-window", action="store_true", help="save charts as PNG files instead of opening windows")
    d.add_argument("--pause", action="store_true", help="wait for Enter between steps")
    d.add_argument("--only", choices=["provider", "claim"], help="show just one model")
    d.set_defaults(f=cmd_demo)
    a = ap.parse_args()
    a.f(a)