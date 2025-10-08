import streamlit as st
import requests
import datetime
from tinydb import TinyDB

st.set_page_config(page_title="Assistente IT", page_icon="🛠️")
st.title("🛠️ Assistente IT Aziendale")

# Form utente
messaggio = st.text_area("Descrivi il problema tecnico:")
categoria = st.selectbox("Categoria", ["VPN", "Office", "Hardware", "Altro"])
priorita = st.radio("Priorità", ["Alta", "Media", "Bassa"])

# Invia richiesta
if st.button("Invia richiesta"):
    payload = {
        "messaggio": messaggio,
        "categoria": categoria,
        "priorita": priorita
    }
    try:
        response = requests.post("http://localhost:8000/ticket", json=payload)
        risposta = response.json().get("risposta", "⚠️ Nessuna risposta ricevuta.")
        st.success("✅ Risposta dell'agente:")
        st.write(risposta)

        # Feedback
        feedback = st.radio("La risposta è stata utile?", ["👍 Sì", "👎 No"])
        with open("log_feedback.csv", "a") as f:
            f.write(f"{datetime.datetime.now()},{messaggio},{categoria},{priorita},{feedback}\n")
    except Exception as e:
        st.error(f"Errore nella richiesta: {e}")

# Cronologia ticket
st.subheader("📜 Cronologia ticket")
try:
    db = TinyDB("tickets_db.json")
    tickets = db.all()
    for item in tickets:
        st.markdown(f"""
        **🕒 {item.get('timestamp', 'N/D')}**
        - Messaggio: {item.get('messaggio', 'N/D')}
        - Categoria: {item.get('categoria', 'N/D')}
        - Priorità: {item.get('priorita', 'N/D')}
        - Stato: {item.get('stato', 'N/D')}
        - Risposta: {item.get('risposta', 'N/D')}
        ---
        """)
except Exception as e:
    st.error(f"Errore nel caricamento cronologia: {e}")

# Pulsante per svuotare la cronologia
if st.button("🗑️ Svuota cronologia ticket"):
    try:
        db.truncate()
        st.success("✅ Cronologia svuotata con successo.")
    except Exception as e:
        st.error(f"Errore durante la cancellazione: {e}")