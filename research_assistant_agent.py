import os
import re
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.llms.openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from llama_index.core.agent import FunctionCallingAgent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def load_api_keys():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

class AgentMemory:
    def __init__(self):
        self.memory = []

    def update(self, user_query: str, agent_response: str):
        self.memory.append({"role": "user", "content": user_query})
        self.memory.append({"role": "assistant", "content": str(agent_response)})

    def get_prompt_context(self) -> str:
        return "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in self.memory])


def generate_subtasks(query: str, llm_planner) -> List[str]:
    planning_template  = """
    You are a task planner for an intelligent agent. Break the following question into high-level subtasks the agent can solve using tools. Use short bullet points.

    Question: {query}
    """
    prompt_template = PromptTemplate(template=planning_template, input_variables=["query"])
    planner_chain  = LLMChain(llm=llm_planner,
                              prompt=prompt_template)

    result = planner_chain.run({"query": query})
    return [subtask.strip("-• ") for subtask in result.strip().split("\n") if subtask.strip()]


def get_doc_tools(file_path: str, name: str, embed_model: OpenAIEmbedding, dataset_path: Optional[str] = None):
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    for doc in documents:
        doc.metadata = {
            "file_name": Path(file_path).name,
            "page_label": doc.metadata.get("page_label"),
            "paper_title": doc.metadata.get("document_title"),
        }

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True) if dataset_path else None
    vector_index = VectorStoreIndex(nodes,
                                    embed_model=embed_model,
                                    vector_store=vector_store)
    summary_index = SummaryIndex(nodes)

    def vector_query(query: str, file_name: Optional[str] = None, page_label: Optional[str] = None) -> str:
        filters = []
        if file_name:
            filters.append({"key": "file_name", "value": file_name})
        if page_label:
            filters.append({"key": "page_label", "value": page_label})

        query_engine = vector_index.as_query_engine(similarity_top_k=5,
                                                    filters=MetadataFilters.from_dicts(filters, condition=FilterCondition.OR)
                                                    )
        return query_engine.query(query)

    def citation_search(query: str, file_name: Optional[str] = None) -> str:
        filters = []
        if file_name:
            filters.append({"key": "file_name", "value": file_name})

        query_engine = vector_index.as_query_engine(similarity_top_k=5,
                                                    filters=MetadataFilters.from_dicts(filters),
                                                    )
        return query_engine.query(f"Find relevant citations: {query}")

    vector_tool = FunctionTool.from_defaults(fn=vector_query,
                                             name=f"vector_tool_{name}",
                                             description=f"Vector-based search over {name} with optional page filter."
                                             )

    summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize",
                                                         use_async=True)

    summary_tool = QueryEngineTool.from_defaults(query_engine=summary_query_engine,
                                                 name=f"summary_tool_{name}",
                                                 description=f"Summarize the content of {name}."
                                                 )

    citation_tool = FunctionTool.from_defaults(fn=citation_search,
                                               name=f"citation_tool_{name}",
                                               description="Search the paper for relevant citations and prior work mentioned across all sections, including Related Work."
                                               )

    return vector_tool, summary_tool, citation_tool


class ResearchAgent:
    def __init__(self, agent_name: str, llm, tools: List, system_prompt: str):
        self.agent_name = agent_name
        self.memory = AgentMemory()
        self.agent = FunctionCallingAgent.from_tools(
            tools=tools,
            llm=llm,
            system_prompt=system_prompt,
            verbose=True,
        )

    def run_query(self, query: str, planner=None):
        steps = generate_subtasks(query, planner) if planner else [query]
        print("\nPlanned Subtasks:")
        for idx, step in enumerate(steps, 1):
            print(f"{idx}. {step}")

        print("\nExecuting subtasks...\n")
        for step in steps:
            print(f"Thought: I need to address the subtask: \"{step}\"")

            # include past memory
            context = self.memory.get_prompt_context()
            task_with_context = f"{context}\n\nSubtask: {step}" if context else step

            print("Action: Calling the appropriate tool using FunctionCallingAgent...")

            response = self.agent.chat(task_with_context)

            print("Observation:\n", response)

            self.memory.update(step, response)

            print(f"{'-'*100}")


def main(query):
    _ = load_api_keys()
    embedding_model = OpenAIEmbedding(model="text-embedding-3-small")
    pdfs_path = Path("pdfs")
    dataset_base = "./dataset/agentic_rag"
    tools = []

    for file in pdfs_path.glob("*.pdf"):
        file_name = re.sub(r"[^a-zA-Z0-9_-]", "_", file.stem)
        print(f"Processing file: {file_name}")

        vector_tool, summary_tool, citation_tool = get_doc_tools(file_path = str(file),
                                                                 name = file_name,
                                                                 embed_model=embedding_model,
                                                                 dataset_path=f"{dataset_base}/{file_name}"
                                                                 )
        tools.extend([vector_tool, summary_tool, citation_tool])

    llm_planner = ChatOpenAI(model="gpt-4",
                             temperature=0)

    llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

    system_prompt = """
    You are a helpful research assistant answering questions using academic PDFs.
    Use the summary tools for summarizing full documents, and vector tools for filtered, specific search.
    You can use filters like file_name and page_label where relevant.
    """

    research_agent = ResearchAgent(agent_name="research_agent",
                                   llm=llm,
                                   tools=tools,
                                   system_prompt=system_prompt)

    research_agent.run_query(query,
                             planner=llm_planner)

if __name__ == "__main__":
    main(query = '''Compare how each academic paper in the collection evaluates or measures bias in large language models.
         For each paper, summarize its key contributions with a focus on bias analysis, and describe the methods or metrics it uses to assess bias.
         After analyzing all papers, determine which one uses the most rigorous or comprehensive approach to bias evaluation, and explain why.''')
