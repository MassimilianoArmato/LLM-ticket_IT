from tinydb import TinyDB
from datetime import datetime

# Inizializza il database
db = TinyDB("tickets_db.json")

def salva_ticket(messaggio, categoria, priorita, risposta, stato="in_attesa"):
    db.insert({
        "messaggio": messaggio,
        "categoria": categoria,
        "priorita": priorita,
        "risposta": risposta,
        "stato": stato,
        "timestamp": datetime.now().isoformat()
    })

def leggi_tutti():
    return db.all()