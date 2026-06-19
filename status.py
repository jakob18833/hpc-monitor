#!/usr/bin/env python3
import re
import subprocess
import sys
from dataclasses import dataclass

try:
    import questionary
    from rich import box
    from rich.console import Console
    from rich.text import Text
    from rich.table import Table
except ImportError:
    print("pip install rich questionary")
    sys.exit(1)

console = Console()


@dataclass
class Node:
    name: str
    state: str
    gpu_type: str = ""
    gpu_total: int = 0
    gpu_alloc: int = 0
    cpu_total: int = 0
    cpu_alloc: int = 0
    mem_total_gb: float = 0
    mem_alloc_gb: float = 0
    partitions: tuple[str, ...] = ()

    @property
    def gpu_free(self): return self.gpu_total - self.gpu_alloc
    @property
    def cpu_free(self): return self.cpu_total - self.cpu_alloc
    @property
    def mem_free_gb(self): return self.mem_total_gb - self.mem_alloc_gb


def parse_mem(s: str) -> float:
    m = re.match(r"([\d.]+)([GMK]?)", s)
    if not m:
        return 0
    val, unit = float(m.group(1)), m.group(2)
    return val if unit == "G" else val / 1024 if unit == "M" else val / (1024 * 1024)


def parse_tres(s: str) -> dict:
    return dict(item.partition("=")[::2] for item in s.split(","))


# Generic CPU-arch / capability tags that aren't a GPU model.
_NON_GPU_TAGS = {"amd", "intel", "genoa", "rome", "turin", "milan", "gpu", "bigmem"}


def gpu_type_from_features(feats: str) -> str:
    tokens = [t for t in feats.split(",") if t and t not in _NON_GPU_TAGS]
    return " ".join(tokens)


def fetch_nodes() -> list[Node]:
    try:
        out = subprocess.check_output(["scontrol", "show", "nodes"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[red]scontrol not found — are you on a SLURM cluster?[/red]")
        sys.exit(1)

    nodes, cur = [], {}
    for line in out.splitlines():
        line = line.strip()
        if m := re.match(r"NodeName=(\S+)", line):
            if cur:
                nodes.append(cur)
            cur = {"name": m.group(1)}
        for key, pattern in [
            ("state",     r"State=(\S+)"),
            ("features",  r"ActiveFeatures=(\S+)"),
            ("partitions",r"Partitions=(\S+)"),
            ("cfg_tres",  r"CfgTRES=(\S+)"),
            ("alloc_tres",r"AllocTRES=(\S+)"),
        ]:
            if m := re.search(pattern, line):
                cur[key] = m.group(1)
    if cur:
        nodes.append(cur)

    result = []
    for d in nodes:
        cfg   = parse_tres(d.get("cfg_tres", ""))
        alloc = parse_tres(d.get("alloc_tres", ""))
        result.append(Node(
            name=d.get("name", ""),
            state=d.get("state", "unknown"),
            gpu_type=gpu_type_from_features(d.get("features", "")),
            gpu_total=int(cfg.get("gres/gpu", 0)),
            gpu_alloc=int(alloc.get("gres/gpu", 0)),
            cpu_total=int(cfg.get("cpu", 0)),
            cpu_alloc=int(alloc.get("cpu", 0)),
            mem_total_gb=parse_mem(cfg.get("mem", "0")),
            mem_alloc_gb=parse_mem(alloc.get("mem", "0")),
            partitions=tuple(p for p in d.get("partitions", "").split(",") if p),
        ))
    return result


def usage_bar(used: float, total: float, width: int = 10) -> Text:
    if total == 0:
        return Text("  N/A  ", style="dim")
    ratio = used / total
    filled = round(ratio * width)
    style = "green" if ratio < 0.5 else "yellow" if ratio < 0.85 else "red"
    bar = Text()
    bar.append("█" * filled, style="black")
    bar.append("█" * (width - filled), style=style)
    return bar


# SLURM state flags that mean a node won't accept a fresh allocation.
_UNAVAILABLE_FLAGS = (
    "down", "drain", "fail", "maint", "reserved", "not_responding",
    "planned", "reboot", "power", "unknown", "invalid", "future",
)


def is_accessible(state: str) -> bool:
    """True only if the node is up and free to take a new job."""
    s = state.lower()
    if "*" in s:  # trailing '*' marks a non-responding node
        return False
    return not any(flag in s for flag in _UNAVAILABLE_FLAGS)


def state_style(state: str) -> str:
    s = state.lower()
    if "idle" in s:   return "green"
    if "mix"  in s:   return "yellow"
    if "alloc" in s:  return "red"
    return "dim"


def show_table(nodes: list[Node]):
    table = Table(box=box.ROUNDED, header_style="bold cyan", show_lines=False)
    table.add_column("Node", style="bold white")
    table.add_column("State")
    table.add_column("GPU type", justify="center")
    table.add_column("GPU free/total", justify="center")
    table.add_column("GPU", justify="center")
    table.add_column("CPU free/total", justify="center")
    table.add_column("CPU", justify="center")
    table.add_column("Mem free/total", justify="center")
    table.add_column("Mem", justify="center")

    for n in nodes:
        table.add_row(
            n.name,
            Text(n.state.split("*")[0], style=state_style(n.state)),
            n.gpu_type or "-",
            f"{n.gpu_free}/{n.gpu_total}",
            usage_bar(n.gpu_alloc, n.gpu_total),
            f"{n.cpu_free}/{n.cpu_total}",
            usage_bar(n.cpu_alloc, n.cpu_total),
            f"{n.mem_free_gb:.0f}/{n.mem_total_gb:.0f} GB",
            usage_bar(n.mem_alloc_gb, n.mem_total_gb),
        )

    console.print(table)
    console.print(f"[dim]{len(nodes)} nodes[/dim]")


def gpu_category(gpu_type: str) -> str:
    t = gpu_type.lower()
    if "mig" in t:
        return "mig"
    if "h100" in t:
        return "h100"
    if "v100" in t:
        return "v100"
    return "other"


def gres_spec(node: Node) -> str:
    # MIG slices need the profile name (e.g. gpu:1g.10gb:1); full GPUs just gpu:1.
    for tok in node.gpu_type.split():
        if re.match(r"\d+g\.\d+gb", tok):
            return f"gpu:{tok}:1"
    return "gpu:1"


def ask_int(message: str, default: int, maximum: int) -> int | None:
    ans = questionary.text(
        message,
        default=str(default),
        validate=lambda x: (x.isdigit() and 1 <= int(x) <= maximum) or f"enter a number 1–{maximum}",
    ).ask()
    return int(ans) if ans else None


def salloc_flow(nodes: list[Node]):
    avail = [n for n in nodes if is_accessible(n.state)]

    # Candidate nodes for each GPU choice (must have a free GPU of that type).
    pools = {
        "h100": [n for n in avail if n.gpu_free > 0 and gpu_category(n.gpu_type) == "h100"],
        "v100": [n for n in avail if n.gpu_free > 0 and gpu_category(n.gpu_type) == "v100"],
        "mig":  [n for n in avail if n.gpu_free > 0 and gpu_category(n.gpu_type) == "mig"],
        "none": avail,
    }

    titles = {"h100": "h100", "v100": "v100", "mig": "mig", "none": "none"}
    choices = []
    for key in ("h100", "v100", "mig", "none"):
        pool = pools[key]
        choices.append(questionary.Choice(
            f"{titles[key]:<5} {len(pool):>3} nodes available",
            value=key,
            disabled=None if pool else "none available",
        ))

    gpu = questionary.select("GPU type:", choices=choices).ask()
    if not gpu:
        return

    # Describe what we want (feature constraint) and let Slurm pick any free
    # matching node — faster and more robust than pinning a single node.
    pool = pools[gpu]

    # Limits = the most a single matching node can give (Slurm places the job on one node).
    max_mem = max(int(n.mem_free_gb) for n in pool)
    max_cpu = max(n.cpu_free for n in pool)
    console.print(
        f"[bold]{len(pool)} matching node(s) free[/bold] "
        f"(up to {max_cpu} CPU / {max_mem} GB on a single node)"
    )

    mem_gb = ask_int(
        f"RAM in GB (max available: {max_mem}):",
        default=min(max_mem, 50),
        maximum=max_mem,
    )
    if mem_gb is None:
        return

    cpus = ask_int(
        f"CPUs (max available: {max_cpu}):",
        default=min(max_cpu, 8),
        maximum=max_cpu,
    )
    if cpus is None:
        return

    minutes = ask_int(
        "Time limit in minutes:",
        default=60,
        maximum=7 * 24 * 60,
    )
    if minutes is None:
        return

    # Partition shared by the matching nodes (preferring non-preempt) — salloc needs one.
    common = set.intersection(*(set(n.partitions) for n in pool))
    partition = min(common, key=lambda p: ("preempt" in p, p)) if common else None

    cmd = ["salloc"]
    if partition:
        cmd.append(f"--partition={partition}")
    cmd.append("--nodes=1")
    if gpu in ("h100", "v100"):
        cmd += [f"--constraint={'v100s' if gpu == 'v100' else 'h100'}", "--gres=gpu:1"]
    elif gpu == "mig":
        cmd.append(f"--gres={gres_spec(max(pool, key=lambda n: n.gpu_free))}")
    cmd += [f"--cpus-per-task={cpus}", f"--mem={mem_gb}G", f"--time={minutes}"]
    console.print("\n[bold]Command:[/bold] " + " ".join(cmd))

    if not questionary.confirm("Run this salloc command?", default=True).ask():
        console.print("[dim]Cancelled.[/dim]")
        return

    subprocess.run(cmd)


def getch() -> str:
    """Read a single keypress without waiting for enter."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def sort_nodes(nodes: list[Node], by: str):
    if by == "gpu":
        # Group by GPU type (GPU nodes first, CPU-only last), then most free GPUs.
        nodes.sort(key=lambda n: (n.gpu_type == "", n.gpu_type, -n.gpu_free, -n.cpu_free))
    else:
        nodes.sort(key=lambda n: n.name)


def main():
    nodes = fetch_nodes()

    sort_by = "name"
    while True:
        sort_nodes(nodes, sort_by)
        console.clear()
        console.print()
        show_table(nodes)
        other = "GPU type" if sort_by == "name" else "node name"
        console.print(f"\n[dim]Press 's' to sort by {other}, any other key to continue…[/dim]")
        if getch().lower() != "s":
            break
        sort_by = "gpu" if sort_by == "name" else "name"

    console.print()
    salloc_flow(nodes)


if __name__ == "__main__":
    main()

