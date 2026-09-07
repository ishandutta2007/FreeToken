"""
FreeToken CLI Entrypoint
"""

import sys
import typer
from rich.console import Console
from rich.table import Table
import uvicorn

app = typer.Typer(
    help="FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution (arXiv:2608.16157)"
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host address to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
    vram: float = typer.Option(8.0, help="Total GPU VRAM limit in GB"),
    workers: int = typer.Option(1, help="Worker count")
):
    """
    Launch the FreeToken OpenAI-compatible serving server.
    """
    console.print(f"[bold green]Starting FreeToken Server on http://{host}:{port}[/bold green]")
    console.print(f"[bold cyan]VRAM Allocation Target:[/bold cyan] {vram} GB")
    console.print("[yellow]Bandwidth-Adaptive $q^\\star$ Execution & Elastic VRAM active.[/yellow]")
    
    from freetoken.server import create_app
    server_app = create_app()
    uvicorn.run(server_app, host=host, port=port)


@app.command()
def benchmark(
    size_mb: int = typer.Option(64, help="Data transfer size in MB for PCIe link benchmark")
):
    """
    Benchmark PCIe host-to-device bandwidth and CPU compute.
    """
    import torch
    from freetoken.policy import HardwareBandwidthProfile

    console.print("[bold blue]Profiling Hardware for Bandwidth-Adaptive Execution...[/bold blue]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Active Device: [green]{device}[/green]")

    bw = HardwareBandwidthProfile.benchmark_pcie(device=device, size_mb=size_mb)
    
    table = Table(title="FreeToken Hardware Profile")
    table.add_column("Resource", style="cyan")
    table.add_column("Measured / Estimated Value", style="magenta")

    table.add_row("Device Type", str(device))
    table.add_row("PCIe H2D Bandwidth", f"{bw:.2f} GB/s")
    table.add_row("Policy", "Bandwidth-Adaptive q* Co-Execution")
    
    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
