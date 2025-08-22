from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .agent import build_agent_with_history, get_session_history

console = Console()

def main():
    session_id = "default"
    cfg = {"configurable": {"session_id": session_id}}

    agent_with_history = build_agent_with_history()

    console.print(Panel(
        "[bold cyan]Agent ready[/bold cyan]. Type 'exit' to quit. "
        "Type '/mem' to inspect memory.",
        style="bold magenta"
    ))

    while True:
        user_input = Prompt.ask("[bold green]you[/bold green]").strip()
        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            console.print("[bold red]Goodbye![/bold red] 👋")
            break

        if user_input == "/mem":
            hist = get_session_history(session_id).messages
            console.print(Panel.fit(f"[bold]Memory has {len(hist)} messages[/bold]"))
            for i, m in enumerate(hist, 1):
                role = getattr(m, "type", m.__class__.__name__)
                content = getattr(m, "content", str(m))
                console.print(f"[dim]{i:02d} {role}[/dim]: {content[:500]}")
            continue

        with console.status("[bold cyan]Thinking...[/bold cyan]"):
            out = agent_with_history.invoke({"input": user_input}, config=cfg)

        console.print(Panel.fit(
            out["output"],
            title="[bold blue]agent[/bold blue]",
            style="bold yellow"
        ))

if __name__ == "__main__":
    main()
