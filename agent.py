import asyncio
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import NodeResult

user_prompt = f"""
Analyze the contents of the 'src' folder and read all files in it.
Respond with the contents of the file that trains a deep learning model.
""".strip()


agent = Agent(
    "openrouter:nvidia/nemotron-3-nano-30b-a3b:free",
)


@agent.tool_plain
def list_files(path: Path) -> str:
    """Returns a list of all files in a directory.
    The 'path' argument must be a directory."""

    # Verify if the directory exists
    if not path.is_dir():
        return "You attempted to read a non-existent folder or a file instead of a folder!"

    files = [
        str(file)
        for file in path.rglob("*")
        if file.is_file() and "__pycache__" not in file.parts
    ]
    return "\n".join(files)


@agent.tool_plain
def read_file(path: Path) -> str:
    """Read a file and return its contents.
    The 'path' argument must be a file."""

    # First verify if the file exists
    if not path.is_file():
        return "You attempted to read a non-existent file!"

    return path.read_text()


async def main():
    all_nodes: list[NodeResult] = []
    async with agent.iter(user_prompt) as agent_run:
        async for node in agent_run:
            all_nodes.append(node)
            print(node)


if __name__ == "__main__":
    asyncio.run(main())
