from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from .agent import agent_executor

console = Console()

def main():
    console.print(Panel("[bold cyan]Agent ready[/bold cyan]. Type 'exit' to quit.", style="bold magenta"))

    while True:
        q = Prompt.ask("[bold green]you[/bold green]")
        if q.lower() in {"exit", "quit"}:
            console.print("[bold red]Goodbye![/bold red] 👋")
            break

        with console.status("[bold cyan]Thinking...[/bold cyan]"):
            out = agent_executor.invoke({"input": q})

        console.print(Panel.fit(out["output"], title="[bold blue]agent[/bold blue]", style="bold yellow"))

if __name__ == "__main__":
    main()
