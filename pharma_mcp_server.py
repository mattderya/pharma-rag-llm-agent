from mcp.server.fastmcp import FastMCP

mcp = FastMCP("pharma-server")


@mcp.tool()
def hello_pharma(name: str) -> str:
    """Say hello to someone in pharma context."""
    return f"Hello {name}, welcome to the pharma MCP server!"


if __name__ == "__main__":
    mcp.run()