import os
import json
import logging
from dotenv import load_dotenv
import requests
import re

import chainlit as cl
from mcp import ClientSession

from semantic_kernel.kernel import Kernel
from azure.core.credentials import AzureKeyCredential

from semantic_kernel.functions import kernel_function
from semantic_kernel.contents import ChatHistory, AuthorRole, ChatMessageContent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat
from semantic_kernel.agents.strategies import SequentialSelectionStrategy, DefaultTerminationStrategy

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchFieldDataType, SearchableField

# ————— Carga configuración —————
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

GITHUB_INSTRUCTIONS = """
You are an expert on GitHub repositories. When answering questions, you **must** use the provided GitHub username to find specific information about that user's repositories, including:

*   Who created the repositories
*   The programming languages used
*   Information found in files and README.md files within those repositories
*   Provide links to each repository referenced in your answers

**Important:** Never perform general searches for repositories. Always use the given GitHub username to find the relevant information. If a GitHub username is not provided, state that you need a username to proceed.
"""

HACKATHON_AGENT = """
You are an AI Agent Hackathon Strategist specializing in recommending winning project ideas.

Your task:
1. Analyze the GitHub activity of users to understand their technical skills
2. Suggest creative AI Agent projects tailored to their expertise. 
3. Focus on projects that align with Microsoft's AI Agent Hackathon prize categories

When making recommendations:
- Base your ideas strictly on the user's GitHub repositories, languages, and tools
- Give suggestions on tools, languages and frameworks to use to build it. 
- Provide detailed project descriptions including architecture and implementation approach
- Explain why the project has potential to win in specific prize categories
- Highlight technical feasibility given the user's demonstrated skills by referencing the specific repositories or languages used.

Formatting your response:
- Provide a clear and structured response that includes:
    - Suggested Project Name
    - Project Description 
    - Potential languages and tools to use
    - Link to each relevant GitHub repository you based your recommendation on

Hackathon prize categories:
- Best Overall Agent ($20,000)
- Best Agent in Python ($5,000)
- Best Agent in C# ($5,000)
- Best Agent in Java ($5,000)
- Best Agent in JavaScript/TypeScript ($5,000)
- Best Copilot Agent using Microsoft Copilot Studio or Microsoft 365 Agents SDK ($5,000)
- Best Azure AI Agent Service Usage ($5,000)
"""

EVENTS_AGENT = """
You are an Event Recommendation Agent specializing in suggesting relevant tech events.

Your task:
1. Review the project idea recommended by the Hackathon Agent
2. Use the search_events function to find relevant events based on the technologies mentioned.
3. NEVER suggest an event where there is not a relevant technology that the user has used.
4. ONLY recommend events that were returned by the search_events function.

When making recommendations:
- IMPORTANT: You must first call the search_events function with appropriate technology keywords from the project
- Only recommend events that were explicitly returned by the search_events function
- Do not make up or suggest events that weren't in the search results
- Construct search queries using specific technologies mentioned (e.g., "Python AI workshop" or "JavaScript hackathon")
- Try multiple search queries if needed to find the most relevant events

For each recommended event:
- Only include events found in the search_events results
- Explain the direct connection between the event and the specific project requirements
- Highlight relevant workshops, sessions, or networking opportunities

Formatting your response:
- Start with "Based on the hackathon project idea, here are relevant events that I found:"
- Only list events that were returned by the search_events function
- For each event, include the exact event details as returned by search_events
- Explain specifically how each event relates to the project technologies

If no relevant events are found, acknowledge this and suggest trying different search terms instead of making up events.
"""

# ————— RAG Plugin —————
class RAGPlugin:
    def __init__(self, search_client):
        self.search_client = search_client

    @kernel_function(name="search_events", description="Searches for relevant events based on a query")
    def search_events(self, query: str) -> str:
        ctx = []
        try:
            for r in self.search_client.search(query, top=5):
                if 'content' in r:
                    ctx.append(f"Event: {r['content']}")
        except Exception as e:
            ctx.append(f"Error Azure Search: {e}")
        try:
            resp = requests.get(f"https://devpost.com/api/hackathons?search={query}", timeout=5)
            if resp.ok:
                for e in resp.json().get('hackathons', [])[:5]:
                    ctx.append(f"Live Event: {e.get('title')} – {e.get('url')}")
        except Exception as e:
            ctx.append(f"Error Devpost API: {e}")
        return "\n\n".join(ctx) if ctx else "No relevant events found."

# ————— Inicializa Azure Search —————
search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
search_key      = os.getenv("AZURE_SEARCH_API_KEY")
index_name      = "event-descriptions"

search_client = SearchClient(endpoint=search_endpoint,
                             index_name=index_name,
                             credential=AzureKeyCredential(search_key))
index_client  = SearchIndexClient(endpoint=search_endpoint,
                                  credential=AzureKeyCredential(search_key))

fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="content", type=SearchFieldDataType.String)
]
idx = SearchIndex(name=index_name, fields=fields)
try:
    index_client.get_index(index_name)
    logger.info("Azure Search index exists.")
except:
    index_client.create_index(idx)
    logger.info("Azure Search index created.")

# Carga descripciones de eventos desde Markdown
base = os.path.dirname(__file__)
md_path = os.path.join(base, "event-descriptions.md")
with open(md_path, "r", encoding="utf-8") as f:
    md = f.read()
docs = [{"id": str(i+1), "content": d.strip()} 
        for i,d in enumerate(md.split("---")) if d.strip()]
if docs:
    try: search_client.delete_documents(documents=[{"id":doc["id"]} for doc in docs])
    except: pass
    search_client.upload_documents(documents=docs)
    logger.info(f"Uploaded {len(docs)} docs to Azure Search.")

# ————— Handlers MCP —————
@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
    logger.info(f"MCP connected: {connection.name}")
    tools = await session.list_tools()
    ts = [{"name":t.name,"description":t.description} for t in tools.tools]
    cl.user_session.set("mcp_tools", {connection.name: ts})
    logger.info(f"Tools for {connection.name}: {[t['name'] for t in ts]}")

@cl.step(type="tool")
async def call_tool(tool_use):
    tool_name, tool_input = tool_use.name, tool_use.input
    mcp = cl.user_session.get("mcp_tools", {})
    mcp_name = next((cn for cn,ts in mcp.items() if any(t["name"]==tool_name for t in ts)), None)
    if not mcp_name:
        return json.dumps({"error": f"Tool {tool_name} not found"})
    session,_ = cl.context.session.mcp_sessions[mcp_name]
    try:
        return await session.call_tool(tool_name, tool_input)
    except Exception as e:
        return json.dumps({"error": str(e)})

@cl.on_chat_start
async def on_chat_start():
    # Carga tus vars de entorno
    endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    api_key    = os.getenv("AZURE_OPENAI_API_KEY")
    api_ver    = os.getenv("AZURE_OPENAI_API_VERSION")

    # 1) Crea el kernel
    kernel = Kernel()

    # 2) Registra el servicio de Azure Chat completions con la firma correcta
    kernel.add_service(
        AzureChatCompletion(
            service_id=       "agent",          
            endpoint=         endpoint,         
            deployment_name=  deployment,       
            api_key=          api_key,          
            api_version=      api_ver           
        )
    )

    settings = kernel.get_prompt_execution_settings_from_service_id("agent")
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # 3) Plugins
    rag = RAGPlugin(search_client)
    kernel.add_plugin(rag, plugin_name="RAG")
    cl.user_session.set("rag_plugin", rag)

    try:
        gh = MCPStdioPlugin(
            name="GitHub",
            description="GitHub MCP Server",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"]
        )
        await gh.connect()
        kernel.add_plugin(gh)
        cl.user_session.set("github_plugin", gh)
        logger.info("GitHub MCP plugin connected")
    except Exception as e:
        logger.error(f"GitHub MCP failed: {e}")

    # 4) Define agents usando siempre service="agent"
    github_agent = ChatCompletionAgent(
        service=kernel.get_service("agent"),
        name="GithubAgent",
        instructions=GITHUB_INSTRUCTIONS,
        plugins=[gh]
    )
    hack_agent = ChatCompletionAgent(
        service=kernel.get_service("agent"),
        name="HackathonAgent",
        instructions=HACKATHON_AGENT
    )
    events_agent = ChatCompletionAgent(
        service=kernel.get_service("agent"),
        name="EventsAgent",
        instructions=EVENTS_AGENT,
        plugins=[rag]
    )

    # 5) Orquesta
    ag_chat = AgentGroupChat(
        agents=[github_agent, hack_agent, events_agent],
        selection_strategy=SequentialSelectionStrategy(initial_agent=github_agent),
        termination_strategy=DefaultTerminationStrategy(maximum_iterations=3)
    )

    # 6) Guarda en sesión
    cl.user_session.set("kernel", kernel)
    cl.user_session.set("settings", settings)
    cl.user_session.set("chat_completion_service", kernel.get_service("agent"))
    cl.user_session.set("chat_history", ChatHistory())
    cl.user_session.set("agent_group_chat", ag_chat)

# ————— on_chat_end —————
@cl.on_chat_end
async def on_chat_end():
    gh = cl.user_session.get("github_plugin")
    if gh: await gh.close()

# ————— Enrutador simple —————
def route_user_input(text: str):
    t = text.lower()
    a=[]
    if re.search(r"github|repo|commit", t): a.append("GithubAgent")
    if re.search(r"hackathon|idea|challenge", t): a.append("HackathonAgent")
    if re.search(r"event|conference|hackathon", t): a.append("EventsAgent")
    return a or ["GithubAgent","HackathonAgent","EventsAgent"]

# ————— on_message —————
@cl.on_message
async def on_message(msg: cl.Message):
    kernel       = cl.user_session.get("kernel")
    service      = cl.user_session.get("chat_completion_service")
    history      = cl.user_session.get("chat_history")
    settings     = cl.user_session.get("settings")
    group_chat   = cl.user_session.get("agent_group_chat")

    history.add_user_message(msg.content)
    agents = route_user_input(msg.content)

    if len(agents)>1:
        await group_chat.add_chat_message(msg.content)
        answer = cl.Message(content=f"Procesando con: {', '.join(agents)}...\n\n")
        await answer.send()
        resp=[]
        try:
            async for c in group_chat.invoke():
                text = f"**{c.name}**: {c.content}\n\n"
                resp.append(text)
                await answer.stream_token(text)
            full = "".join(resp)
            history.add_assistant_message(full)
            answer.content=full
            await answer.update()
        except Exception as e:
            await answer.stream_token(f"❌ Error: {e}")
    else:
        agent = agents[0]
        answer = cl.Message(content=f"Procesando con {agent}...\n\n")
        await answer.send()
        try:
            async for m in service.get_streaming_chat_message_content(
                chat_history=history,
                user_input=msg.content,
                settings=settings,
                kernel=kernel
            ):
                if m.content: await answer.stream_token(m.content)
                if isinstance(m, FunctionCallContent):
                    await answer.stream_token(f"\nCalling function {m.function_name}({m.arguments})\n")
                if isinstance(m, FunctionResultContent):
                    await answer.stream_token(f"\nFunction result: {m.content}\n")
            history.add_assistant_message(answer.content)
            await answer.update()
        except Exception as e:
            await answer.stream_token(f"❌ Error: {e}")

