import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["pharma_mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Server'a baglandim")

            tools = await session.list_tools()
            print("\nMevcut tool'lar:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            result = await session.call_tool(
                "hello_pharma",
                arguments={"name": "Matt"}
            )

            print("\nTool sonucu:")
            for content in result.content:
                print(f"  {content.text}")


if __name__ == "__main__":
    asyncio.run(main())