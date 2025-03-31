# keep these 3 lines. from https://stackoverflow.com/questions/76958817/streamlit-your-system-has-an-unsupported-version-of-sqlite3-chroma-requires-sq
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3') 


import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from crewai_tools import *





load_dotenv()

def create_lmstudio_llm(model, temperature):
    api_base = os.getenv('LMSTUDIO_API_BASE')
    os.environ["OPENAI_API_KEY"] = "lm-studio"
    os.environ["OPENAI_API_BASE"] = api_base
    if api_base:
        return ChatOpenAI(openai_api_key='lm-studio', openai_api_base=api_base, temperature=temperature)
    else:
        raise ValueError("LM Studio API base not set in .env file")

def create_openai_llm(model, temperature):
    safe_pop_env_var('OPENAI_API_KEY')
    safe_pop_env_var('OPENAI_API_BASE')
    load_dotenv(override=True)
    api_key = os.getenv('OPENAI_API_KEY')
    api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1/')
    if api_key:
        return ChatOpenAI(openai_api_key=api_key, openai_api_base=api_base, model_name=model, temperature=temperature)
    else:
        raise ValueError("OpenAI API key not set in .env file")

def create_groq_llm(model, temperature):
    api_key = os.getenv('GROQ_API_KEY')
    if api_key:
        return ChatGroq(groq_api_key=api_key, model_name=model, temperature=temperature)
    else:
        raise ValueError("Groq API key not set in .env file")

def create_anthropic_llm(model, temperature):
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        return ChatAnthropic(anthropic_api_key=api_key, model_name=model, temperature=temperature)
    else:
        raise ValueError("Anthropic API key not set in .env file")

def safe_pop_env_var(key):
    try:
        os.environ.pop(key)
    except KeyError:
        pass
        
LLM_CONFIG = {
    "OpenAI": {
        "create_llm": create_openai_llm
    },
    "Groq": {
        "create_llm": create_groq_llm
    },
    "LM Studio": {
        "create_llm": create_lmstudio_llm
    },
    "Anthropic": {
        "create_llm": create_anthropic_llm
    }
}

def create_llm(provider_and_model, temperature=0.1):
    provider, model = provider_and_model.split(": ")
    create_llm_func = LLM_CONFIG.get(provider, {}).get("create_llm")
    if create_llm_func:
        return create_llm_func(model, temperature)
    else:
        raise ValueError(f"LLM provider {provider} is not recognized or not supported")

def load_agents():
    agents = [
        
Agent(
    role="Estagi\u00e1rio de Varredura de Not\u00edcias",
    backstory="Voc\u00ea est\u00e1 come\u00e7ando sua carreira em uma ag\u00eancia de not\u00edcias. Sua miss\u00e3o \u00e9 vasculhar os conte\u00fados na web e extrair rapidamente os dados essenciais de cada not\u00edcia. Esse esbo\u00e7o ser\u00e1 utilizado pelo redator para criar um resumo final.",
    goal="Voc\u00ea recebeu uma lista de links de not\u00edcias. Sua tarefa \u00e9 acessar cada link utilizando a ferramenta ScrapeWebsiteTool e extrair as informa\u00e7\u00f5es essenciais: t\u00edtulo, pontos principais e o link do artigo. Retorne um esbo\u00e7o com essas informa\u00e7\u00f5es para servir de base aos pr\u00f3ximos agentes.",
    allow_delegation=False,
    verbose=False,
    tools=[ScrapeWebsiteTool()],
    llm=create_llm("OpenAI: gpt-4o-mini", 0.1)
)
            ,
        
Agent(
    role="Redator de Not\u00edcias S\u00eanior",
    backstory="Voc\u00ea \u00e9 um redator experiente, capaz de transformar um esbo\u00e7o em um resumo claro, conciso e informativo. Independente do idioma original do artigo, seu resumo deve ser produzido em portugu\u00eas.",
    goal="Utilize o esbo\u00e7o produzido pelo estagi\u00e1rio para elaborar um resumo final detalhado de cada not\u00edcia. \n        Para cada artigo, o resumo deve incluir:\n        - T\u00edtulo do artigo\n        - Resumo final, escrito em um \u00fanico par\u00e1grafo, com os principais fatos\n        - Lista de palavras-chave relevantes\n        - Link original do artigo\nCaso necess\u00e1rio, utilize a ferramenta docs_scrape_website para confirmar detalhes do conte\u00fado.",
    allow_delegation=False,
    verbose=False,
    tools=[],
    llm=create_llm("OpenAI: gpt-4o-mini", 0.1)
)
            ,
        
Agent(
    role="Editor Chefe de Not\u00edcias",
    backstory="Como editor chefe, voc\u00ea garante a qualidade final dos conte\u00fados publicados. Sua an\u00e1lise cr\u00edtica far\u00e1 com que os resumos sejam diretos e perfeitamente formatados para publica\u00e7\u00e3o, sem perder a clareza e objetividade necess\u00e1rias para a comunica\u00e7\u00e3o r\u00e1pida.",
    goal="Revise e edite os resumos produzidos pelo redator, garantindo que estejam claros, concisos e acompanhados de todas as informa\u00e7\u00f5es essenciais (t\u00edtulo, resumo, palavras-chave e link). Certifique-se que o texto final seja escrito em um \u00fanico par\u00e1grafo e atenda aos padr\u00f5es editoriais exigidos.",
    allow_delegation=False,
    verbose=False,
    tools=[],
    llm=create_llm("OpenAI: gpt-4o-mini", 0.1)
)
            
    ]
    return agents

def load_tasks(agents):
    tasks = [
        
Task(
    description="Voc\u00ea recebeu a seguinte lista de links de not\u00edcias {links}. Para cada link, utilize a ferramenta ScrapeWebTool\n        para acessar o artigo e extrair as seguintes informa\u00e7\u00f5es:\n          - T\u00edtulo da not\u00edcia\n          - Pontos principais ou resumo inicial\n          - Link original\n        Retorne um esbo\u00e7o contendo essas informa\u00e7\u00f5es no seguinte formato:\n        T\u00edtulo: <t\u00edtulo do artigo>\n        Conte\u00fado: <pontos principais ou resumo inicial>\n        Link: <URL do artigo>",
    expected_output="Formato esperado:\n        T\u00edtulo: <t\u00edtulo do artigo>\n        Conte\u00fado: <pontos principais ou resumo inicial>\n        Link: <URL do artigo>",
    agent=next(agent for agent in agents if agent.role == "Estagi\u00e1rio de Varredura de Not\u00edcias"),
    async_execution=False
)
            ,
        
Task(
    description="Utilize o esbo\u00e7o inicial produzido pelo estagi\u00e1rio para elaborar um resumo final para cada not\u00edcia. \n        Certifique-se de que o resumo contenha:\n          - T\u00edtulo do artigo\n          - Resumo final, conciso e informativo (em um \u00fanico par\u00e1grafo)\n          - Lista de palavras-chave relevantes\n          - Link original do artigo\n        Caso necess\u00e1rio, confirme detalhes utilizando a ferramenta docs_scrape_tool. Os links s\u00e3o {links}.",
    expected_output="Formato esperado:\n        T\u00edtulo: <t\u00edtulo do artigo>\n        Resumo: <resumo final em um \u00fanico par\u00e1grafo>\n        Palavras-chave: <lista de palavras-chave separadas por v\u00edrgula>\n        Link: <URL do artigo>",
    agent=next(agent for agent in agents if agent.role == "Redator de Not\u00edcias S\u00eanior"),
    async_execution=False
)
            ,
        
Task(
    description="Revise o resumo final produzido pelo redator para cada not\u00edcia e assegure que:\n          - O t\u00edtulo, resumo, palavras-chave e link estejam presentes e bem formatados\n          - O resumo esteja escrito em um \u00fanico par\u00e1grafo, com clareza e concis\u00e3o\n        Realize as corre\u00e7\u00f5es necess\u00e1rias para que o conte\u00fado atenda ao padr\u00e3o editorial.",
    expected_output="Formato final esperado:\n        T\u00edtulo: <t\u00edtulo do artigo>\n        Resumo: <resumo final revisado, em um \u00fanico par\u00e1grafo>\n        Palavras-chave: <lista de palavras-chave separadas por v\u00edrgula>\n        Link: <URL do artigo>",
    agent=next(agent for agent in agents if agent.role == "Editor Chefe de Not\u00edcias"),
    async_execution=False
)
            
    ]
    return tasks

def main():
    st.title("Resumidor de links")

    agents = load_agents()
    tasks = load_tasks(agents)
    crew = Crew(
        agents=agents, 
        tasks=tasks, 
        process="sequential", 
        verbose=True, 
        memory=True, 
        cache=True, 
        max_rpm=1000,
        
    )

    links = st.text_input("Links")

    placeholders = {
        "links": links
    }
    with st.spinner("Running crew..."):
        try:
            result = crew.kickoff(inputs=placeholders)
            with st.expander("Final output", expanded=True):
                if hasattr(result, 'raw'):
                    st.write(result.raw)                
            with st.expander("Full output", expanded=False):
                st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
