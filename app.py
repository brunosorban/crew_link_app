# Keep these 3 lines from https://stackoverflow.com/questions/76958817/streamlit-your-system-has-an-unsupported-version-of-sqlite3-chroma-requires-sq
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os

import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai_tools import *
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Set environment variables
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# **Define your LLM creation functions**
def create_openai_llm(model, temperature):
    # Accessing API key and base from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/")

    if api_key:
        return ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=api_base,
            model_name=model,
            temperature=temperature,
        )
    else:
        raise ValueError("OpenAI API key not set in environment variables")


LLM_CONFIG = {"OpenAI": {"create_llm": create_openai_llm}}


def create_llm(provider_and_model, temperature=0.1):
    try:
        provider, model = provider_and_model.split(": ")
    except ValueError:
        raise ValueError("provider_and_model must be in the format 'Provider: Model'")
    create_llm_func = LLM_CONFIG.get(provider, {}).get("create_llm")
    if create_llm_func:
        return create_llm_func(model, temperature)
    else:
        raise ValueError(
            f"LLM provider '{provider}' is not recognized or not supported"
        )


def load_agents():
    agents = [
        Agent(
            role="Estagiário de Varredura de Notícias",
            backstory="Você está começando sua carreira em uma agência de notícias. Sua missão é vasculhar os conteúdos na web e extrair rapidamente os dados essenciais de cada notícia. Esse esboço será utilizado pelo redator para criar um resumo final.",
            goal="Você recebeu uma lista de links de notícias. Sua tarefa é acessar cada link utilizando a ferramenta ScrapeWebsiteTool e extrair as informações essenciais: título, pontos principais e o link do artigo. Retorne um esboço com essas informações para servir de base aos próximos agentes.",
            allow_delegation=False,
            verbose=False,
            tools=[ScrapeWebsiteTool()],
            llm=create_llm("OpenAI: gpt-4o-mini", 0.1),
        ),
        Agent(
            role="Redator de Notícias Sênior",
            backstory="Você é um redator experiente, capaz de transformar um esboço em um resumo claro, conciso e informativo. Independente do idioma original do artigo, seu resumo deve ser produzido em português.",
            goal="Utilize o esboço produzido pelo estagiário para elaborar um resumo final detalhado de cada notícia. \n        Para cada artigo, o resumo deve incluir:\n        - Título do artigo\n        - Resumo final, escrito em um único parágrafo, com os principais fatos\n        - Lista de palavras-chave relevantes\n        - Link original do artigo\nCaso necessário, utilize a ferramenta docs_scrape_website para confirmar detalhes do conteúdo.",
            allow_delegation=False,
            verbose=False,
            tools=[],
            llm=create_llm("OpenAI: gpt-4o-mini", 0.1),
        ),
        Agent(
            role="Editor Chefe de Notícias",
            backstory="Como editor chefe, você garante a qualidade final dos conteúdos publicados. Sua análise crítica fará com que os resumos sejam diretos e perfeitamente formatados para publicação, sem perder a clareza e objetividade necessárias para a comunicação rápida.",
            goal="Revise e edite os resumos produzidos pelo redator, garantindo que estejam claros, concisos e acompanhados de todas as informações essenciais (título, resumo, palavras-chave e link). Certifique-se que o texto final seja escrito em um único parágrafo e atenda aos padrões editoriais exigidos.",
            allow_delegation=False,
            verbose=False,
            tools=[],
            llm=create_llm("OpenAI: gpt-4o-mini", 0.1),
        ),
    ]
    return agents


def load_tasks(agents):
    tasks = [
        Task(
            description=(
                "Você recebeu a seguinte lista de links de notícias {links}. Para cada link, utilize a ferramenta ScrapeWebTool\n"
                "para acessar o artigo e extrair as seguintes informações:\n"
                "  - Título da notícia\n"
                "  - Pontos principais ou resumo inicial\n"
                "  - Link original\n"
                "Retorne um esboço contendo essas informações no seguinte formato:\n"
                "Título: <título do artigo>\n"
                "Conteúdo: <pontos principais ou resumo inicial>\n"
                "Link: <URL do artigo>"
            ),
            expected_output=(
                "Formato esperado:\n"
                "Título: <título do artigo>\n"
                "Conteúdo: <pontos principais ou resumo inicial>\n"
                "Link: <URL do artigo>"
            ),
            agent=next(
                agent
                for agent in agents
                if agent.role == "Estagiário de Varredura de Notícias"
            ),
            async_execution=False,
        ),
        Task(
            description=(
                "Utilize o esboço inicial produzido pelo estagiário para elaborar um resumo final para cada notícia. \n"
                "Certifique-se de que o resumo contenha:\n"
                "  - Título do artigo\n"
                "  - Resumo final, conciso e informativo (em um único parágrafo)\n"
                "  - Lista de palavras-chave relevantes\n"
                "  - Link original do artigo\n"
                "Caso necessário, confirme detalhes utilizando a ferramenta docs_scrape_tool. Os links são {links}."
            ),
            expected_output=(
                "Formato esperado:\n"
                "Título: <título do artigo>\n"
                "Resumo: <resumo final em um único parágrafo>\n"
                "Palavras-chave: <lista de palavras-chave separadas por vírgula>\n"
                "Link: <URL do artigo>"
            ),
            agent=next(
                agent for agent in agents if agent.role == "Redator de Notícias Sênior"
            ),
            async_execution=False,
        ),
        Task(
            description=(
                "Revise o resumo final produzido pelo redator para cada notícia e assegure que:\n"
                "  - O título, resumo, palavras-chave e link estejam presentes e bem formatados\n"
                "  - O resumo esteja escrito em um único parágrafo, com clareza e concisão\n"
                "Realize as correções necessárias para que o conteúdo atenda ao padrão editorial."
            ),
            expected_output=(
                "Formato final esperado:\n"
                "Título: <título do artigo>\n"
                "Resumo: <resumo final revisado, em um único parágrafo>\n"
                "Palavras-chave: <lista de palavras-chave separadas por vírgula>\n"
                "Link: <URL do artigo>"
            ),
            agent=next(
                agent for agent in agents if agent.role == "Editor Chefe de Notícias"
            ),
            async_execution=False,
        ),
    ]
    return tasks


def main():
    st.title("Resumidor de links")

    # **Load Agents and Tasks**
    agents = load_agents()
    tasks = load_tasks(agents)

    try:
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process="sequential",
            verbose=True,
            memory=True,
            cache=True,
            max_rpm=1000,
        )
    except Exception as e:
        st.error(f"Error initializing Crew: {str(e)}")
        st.stop()  # Stop further execution if Crew initialization fails

    links = st.text_input("Links")

    # **Ensure Links Are Provided**
    if not links:
        st.info("Please enter one or more links to proceed.")
        st.stop()

    placeholders = {"links": links}
    with st.spinner("Running crew..."):
        try:
            result = crew.kickoff(inputs=placeholders)
            with st.expander("Final output", expanded=True):
                if hasattr(result, "raw"):
                    st.write(result.raw)
            with st.expander("Full output", expanded=False):
                st.write(result)
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")


if __name__ == "__main__":
    main()
