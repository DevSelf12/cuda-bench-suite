"""JSON report generator and comparison tool."""

import json
from pathlib import Path
from typing import Any


def save_json(data: dict, path: str) -> None:
    """Save results as formatted JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, default=str))


def compare_results(results: list[dict], fmt: str = "text") -> str:
    """Compare multiple benchmark result sets."""
    if fmt == "text":
        return _compare_text(results)
    elif fmt == "html":
        return _compare_html(results)
    elif fmt == "csv":
        return _compare_csv(results)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def _compare_text(results: list[dict]) -> str:
    """Plain text comparison table."""
    lines = ["=" * 80]
    lines.append("GPU BENCHMARK COMPARISON")
    lines.append("=" * 80)

    for r in results:
        gpu = r.get("gpu", {}).get("name", "Unknown")
        lines.append(f"\n--- {gpu} ---")
        for bench in r.get("benchmarks", []):
            name = bench.get("name", "?")
            if name == "matmul":
                for item in bench.get("results", []):
                    lines.append(
                        f"  matmul {item['size']}x{item['size']} ({item['precision']}): "
                        f"{item.get('tflops', '?')} TFLOPS, {item.get('avg_ms', '?')} ms"
                    )
            elif name == "bandwidth":
                for item in bench.get("results", []):
                    lines.append(
                        f"  bandwidth {item['direction']}: {item.get('bandwidth_gbps', '?')} GB/s"
                    )

    output = "\n".join(lines)
    print(output)
    return output


def _compare_html(results: list[dict]) -> str:
    """HTML comparison table."""
    html = ["<html><head><style>",
            "body{font-family:monospace;background:#1a1a2e;color:#e0e0e0;padding:20px}",
            "table{border-collapse:collapse;width:100%}",
            "th,td{border:1px solid #333;padding:8px;text-align:center}",
            "th{background:#16213e;color:#0f3460}",
            "tr:nth-child(even){background:#16213e}",
            "</style></head><body>",
            "<h1>GPU Benchmark Comparison</h1>",
            "<table><tr><th>GPU</th><th>Benchmark</th><th>Config</th><th>Result</th></tr>"]

    for r in results:
        gpu = r.get("gpu", {}).get("name", "Unknown")
        for bench in r.get("benchmarks", []):
            name = bench.get("name", "?")
            if name == "matmul":
                for item in bench.get("results", []):
                    html.append(
                        f"<tr><td>{gpu}</td><td>matmul</td>"
                        f"<td>{item['size']}x{item['size']} {item['precision']}</td>"
                        f"<td>{item.get('tflops', '?')} TFLOPS</td></tr>"
                    )

    html.extend(["</table></body></html>"])
    output = "\n".join(html)
    return output


def _compare_csv(results: list[dict]) -> str:
    """CSV comparison."""
    lines = ["gpu,benchmark,config,result,unit"]
    for r in results:
        gpu = r.get("gpu", {}).get("name", "Unknown")
        for bench in r.get("benchmarks", []):
            name = bench.get("name", "?")
            if name == "matmul":
                for item in bench.get("results", []):
                    lines.append(
                        f"{gpu},matmul,{item['size']}x{item['size']}_{item['precision']},"
                        f"{item.get('tflops', 0)},TFLOPS"
                    )
    return "\n".join(lines)
