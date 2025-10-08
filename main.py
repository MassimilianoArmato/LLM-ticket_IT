from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from db import salva_ticket
import logging
import os
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 📦 Caricamento modello Hugging Face ottimizzato
llm_pipeline = None
try:
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16  # rimuovi se non hai GPU
    )
    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    test_output = llm_pipeline("Ciao, come posso aiutarti?", max_new_tokens=100)[0]["generated_text"]
    logger.info(f"✅ Test modello completato: {test_output}")

except Exception as e:
    import traceback
    logger.error("❌ Errore nel caricamento del modello:")
    traceback.print_exc()
    llm_pipeline = None

# 📚 KB FAISS
qa_chain = None
retrieval_tool = None
try:
    if os.path.exists("faiss_index"):
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db_vector = FAISS.load_local(
            "faiss_index",
            embedding_model,
            allow_dangerous_deserialization=True
        )
        def run_qa(query):
            return qa_chain.run(query)

        qa_chain = RetrievalQA.from_chain_type(llm=run_qa, retriever=db_vector.as_retriever())
        retrieval_tool = Tool(
            name="search_docs",
            func=run_qa,
            description="Cerca informazioni tecniche nei documenti aziendali"
        )
        logger.info("✅ KB FAISS e tool caricati.")
    else:
        logger.warning("⚠️ Nessuna KB FAISS trovata.")
except Exception as e:
    logger.error(f"❌ Errore nel caricamento della KB: {e}")

# 🤖 Agente LangChain
agent = None
try:
    tools = [retrieval_tool] if retrieval_tool else []
    if tools:
        def llm_wrapper(prompt):
            return llm_pipeline(prompt, max_new_tokens=100)[0]["generated_text"]

        agent = initialize_agent(tools=tools, llm=llm_wrapper, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        logger.info("✅ Agente LangChain inizializzato.")
    else:
        logger.warning("⚠️ Nessun tool disponibile per l'agente.")
except Exception as e:
    logger.error(f"❌ Errore nell'inizializzazione dell'agente: {e}")

# 🧾 Schema input
class Ticket(BaseModel):
    messaggio: str
    categoria: str
    priorita: str

# 🧠 Prompt fallback
template = """Messaggio: {messaggio}
Risposta:"""
prompt = PromptTemplate.from_template(template)

# 📩 Endpoint API
@app.post("/ticket")
async def ricevi_ticket(ticket: Ticket):
    messaggio = ticket.messaggio
    categoria = ticket.categoria
    priorita = ticket.priorita

    if llm_pipeline is None:
        logger.warning("⚠️ LLM non disponibile.")
        return {"risposta": "⚠️ Il sistema non è pronto. Riprova più tardi."}

    try:
        if agent:
            risposta = agent.run(messaggio)
        else:
            final_prompt = prompt.format(messaggio=messaggio)
            risposta = llm_pipeline(final_prompt, max_new_tokens=100)[0]["generated_text"]
    except Exception as e:
        logger.error(f"❌ Errore durante la generazione della risposta: {e}")
        risposta = "⚠️ Errore interno. Riprova più tardi."

    salva_ticket(messaggio, categoria, priorita, risposta, stato="chiuso")
    return {"risposta": risposta}