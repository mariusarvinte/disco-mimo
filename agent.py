import asyncio
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import NodeResult

user_prompt = """
Analyze the contents of the 'src' folder and read all files in it.
Identify the file responsible for training a deep learning model.
Identiy two possible optimizations that could be applied to the
training code to make it faster without sacrificing numerical precision or performance.
Write them to 'PLAN.md'.
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


@agent.tool_plain
def write_plan(path: Path, contents: str) -> str:
    """Write a plan to an .md file"""

    # Verify if the to-be-written file is an .md file that doesn't already exist
    if path.is_file():
        return "File already exists!"

    if path != "PLAN.md":
        return "You can only write to the PLAN.md file!"

    path.write_text(contents)
    return "File was written successfully"


async def main():
    all_nodes: list[NodeResult] = []
    async with agent.iter(user_prompt) as agent_run:
        async for node in agent_run:
            all_nodes.append(node)
            print(node)


if __name__ == "__main__":
    asyncio.run(main())
