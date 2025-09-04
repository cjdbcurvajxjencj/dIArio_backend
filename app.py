import os
import threading
import time
import json
import hashlib
from pathlib import Path
from flask import Flask, request, jsonify
# from pyngrok import ngrok  # <-- RIMOSSO: Non necessario in un ambiente di produzione come Render.
import google.generativeai as genai
from google.api_core import exceptions
import random
# from kaggle_secrets import UserSecretsClient  # <-- RIMOSSO: I segreti non vengono più gestiti da Kaggle.
from flask_cors import CORS
import tempfile
from datetime import datetime, timezone, timedelta
from pydub import AudioSegment
import io
import subprocess

# --- CONFIGURAZIONE ---
# Il blocco di configurazione per NGROK e Kaggle è stato rimosso.
# Le chiavi API ora vengono gestite direttamente dalla logica dell'applicazione
# e dalle variabili d'ambiente fornite dalla piattaforma di hosting (Render).

app = Flask(__name__)
CORS(app)

# --- IMPORTANTE: Questo percorso viene mantenuto identico. ---
# Su Render, creerai un "Persistent Disk" e lo monterai esattamente
# su questo percorso. In questo modo, il codice non necessita di modifiche
# e i tuoi file JSON saranno salvati in modo permanente.
JOBS_DIR = Path("/kaggle/working/jobs")
JOBS_DIR.mkdir(exist_ok=True, parents=True)


# --- FUNZIONI HELPER ---
# NESSUNA MODIFICA: Questa funzione è invariata.
def get_user_hash(api_key):
    return hashlib.sha256(api_key.encode('utf-8')).hexdigest()

# --- FUNZIONE DI ELABORAZIONE GEMINI ---
# NESSUNA MODIFICA: La logica di elaborazione principale è rimasta assolutamente invariata.
# La chiave API di Gemini (`api_key`) viene ancora passata per ogni singolo job,
# preservando il modello "bring your own key" originale.
def transcribe_and_summarize_task(lesson_id, user_hash, api_key, raw_input_path_str, subject, transcription_model_name, summary_model_name):
    job_file = JOBS_DIR / user_hash / f"{lesson_id}.json"
    print(f"[{lesson_id}] Inizio elaborazione per utente {user_hash[:8]}... (Trascrizione: {transcription_model_name}, Riassunto: {summary_model_name})")
    
    raw_input_file = Path(raw_input_path_str)
    sanitized_temp_file = None
    try:
        # --- PASSAGGIO 1: RIPARAZIONE E STANDARDIZZAZIONE FILE-TO-FILE ---
        print(f"[{lesson_id}] Avvio passaggio di riparazione da file a file...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_f:
            sanitized_temp_file = Path(temp_f.name)

        repair_command = [
            'ffmpeg', '-y', '-analyzeduration', '20M', '-probesize', '20M',
            '-i', str(raw_input_file),
            '-vn', '-acodec', 'libmp3lame', '-b:a', '192k', '-f', 'mp3',
            str(sanitized_temp_file)
        ]
        
        result = subprocess.run(repair_command, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            print(f"ERRORE FFMPEG (Riparazione): {result.stderr}")
            raise RuntimeError("Fase 1 fallita: impossibile riparare il file audio.")

        # --- PASSAGGIO 2: CONTROLLO DI INTEGRITÀ ---
        print(f"[{lesson_id}] File audio intermedio salvato. Avvio controllo di integrità...")

        if not sanitized_temp_file.exists() or sanitized_temp_file.stat().st_size < 1024:
            raise RuntimeError("Fase 2 fallita: il processo di riparazione ha generato un file vuoto o troppo piccolo.")

        probe_command = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(sanitized_temp_file)
        ]
        
        result = subprocess.run(probe_command, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            print(f"ERRORE FFPROBE (Verifica): {result.stderr}")
            raise RuntimeError("Fase 2 fallita: il file audio riparato è ancora illeggibile.")
        
        print(f"[{lesson_id}] Controllo di integrità superato. Durata rilevata: {result.stdout.strip()}s.")
        
        # La configurazione di Gemini avviene qui, usando la chiave specifica dell'utente per questo job.
        # Comportamento INVARIATO.
        genai.configure(api_key=api_key)
        transcription_model = genai.GenerativeModel(transcription_model_name)
        summary_model = genai.GenerativeModel(summary_model_name)

        # --- PASSAGGIO 3: TRASCRIZIONE A BLOCCHI ---
        print(f"[{lesson_id}] Divisione del file audio MP3 in blocchi...")
        
        audio = AudioSegment.from_file(sanitized_temp_file, format="mp3")

        CHUNK_LENGTH_MS = 15 * 60 * 1000
        transcripts = []
        total_chunks = (len(audio) // CHUNK_LENGTH_MS) + (1 if len(audio) % CHUNK_LENGTH_MS > 0 else 0)

        for i in range(total_chunks):
            start_ms = i * CHUNK_LENGTH_MS
            end_ms = start_ms + CHUNK_LENGTH_MS
            chunk = audio[start_ms:end_ms]
            
            current_chunk_index = i + 1
            print(f"[{lesson_id}] Elaborazione blocco {current_chunk_index}/{total_chunks}...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_chunk_file:
                chunk.export(temp_chunk_file.name, format="mp3")
                temp_chunk_path = Path(temp_chunk_file.name)

            audio_file_resource = None
            try:
                response = None
                max_retries = 10
                for attempt in range(max_retries):
                    try:
                        print(f"[{lesson_id}] Caricamento blocco {current_chunk_index} a Gemini (Tentativo {attempt + 1})...")
                        audio_file_resource = genai.upload_file(path=temp_chunk_path, mime_type="audio/mpeg")

                        while audio_file_resource.state.name == "PROCESSING":
                            time.sleep(10)
                            audio_file_resource = genai.get_file(audio_file_resource.name)

                        if audio_file_resource.state.name == "FAILED":
                            raise ValueError("Elaborazione blocco audio fallita su Gemini.")
                        
                        if not transcripts:
                            prompt = """Sei un assistente IA specializzato nella trascrizione di lezioni accademiche, incaricato di produrre un testo fedele e leggibile. Il tuo obiettivo è una trascrizione completa che subisce solo una leggerissima revisione stilistica.

**Regole Fondamentali:**
1.  **Trascrizione Completa (Priorità Massima):** Trascrivi **ogni parola** pronunciata dal docente. Non omettere frasi, concetti o esempi. Le ripetizioni di concetti o intere frasi sono importanti e **devono essere mantenute** perché fanno parte dello stile espositivo. L'output **non è un riassunto**.
2.  **Revisione Leggera e Conservativa:**
    * **Cosa Rimuovere:** Elimina **solo ed esclusivamente** le seguenti distrazioni verbali:
        * Interiezioni e suoni di esitazione (es: 'ehm', 'uhm').
        * Ripetizioni immediate e involontarie della stessa parola (es. "il il libro" diventa "il libro").
        * Intercalari usati chiaramente come riempitivo e non per enfasi (es. l'abuso di "quindi", "cioè", "diciamo"). Usali con parsimonia solo se necessari per il flusso del discorso.
    * **Cosa Mantenere:** Mantieni la struttura originale delle frasi. Correggi solo le false partenze evidenti o gli errori grammaticali palesi, ma **non riformulare le frasi** per renderle più eleganti. L'autenticità del parlato è importante.
3.  **Focus sul Docente:** Ignora completamente rumori di fondo, brusii, colpi di tosse o domande degli studenti. Trascrivi solo la voce del docente principale.
4.  **Formattazione:**
    * Usa una punteggiatura accurata e suddividi il testo in paragrafi logici.
    * Formatta le formule matematiche usando la sintassi LaTeX (es. $E=mc^2$).
5.  **Output Diretto:** Restituisci **solo ed esclusivamente** il testo della trascrizione. Nessuna introduzione, nessun commento."""
                        else:
                            previous_context = transcripts[-1][-250:]
                            prompt = f"""Stai continuando una trascrizione accademica. Il tuo compito è trascrivere il nuovo segmento audio, collegandoti in modo fluido al contesto fornito e seguendo le stesse regole del prompt iniziale.

**CONTESTO (ULTIMA PARTE DELLA TRASCRIZIONE PRECEDENTE - NON RIPETERLO):**
---
...{previous_context}
---

**REGOLE CHIAVE DA RICORDARE:**
1. **Continuità:** Non ripetere il contesto. Inizia a trascrivere dal punto esatto in cui il nuovo audio riprende.
2. **Stile:** Mantieni la stessa revisione leggera (rimuovi 'ehm', 'uhm', ecc.). Non includere timestamp.
3. **Output Diretto:** Restituisci solo il testo della nuova trascrizione, senza commenti o introduzioni."""

                        response = transcription_model.generate_content([prompt, audio_file_resource], request_options={'timeout': 1800})
                        break 
                    except exceptions.ResourceExhausted as e:
                        wait_time = (2 ** attempt) * 5 + random.uniform(0, 1)
                        print(f"[{lesson_id}] Rate limit superato per trascrizione (tentativo {attempt + 1}/{max_retries}). Dettagli API: {e}. Attendo {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    except Exception as e:
                        print(f"[{lesson_id}] Errore API imprevisto durante la trascrizione: {e}")
                        raise
                
                if response is None:
                    raise RuntimeError(f"Impossibile ottenere la trascrizione dopo {max_retries} tentativi.")
                
                transcripts.append(response.text)
                print(f"[{lesson_id}] Trascrizione blocco {current_chunk_index} completata.")

            finally:
                if temp_chunk_path and temp_chunk_path.exists(): temp_chunk_path.unlink()
                if audio_file_resource:
                    try:
                        genai.delete_file(audio_file_resource.name)
                    except Exception as delete_error:
                        print(f"[{lesson_id}] Attenzione: impossibile eliminare file del blocco da Gemini. Errore: {delete_error}")
            
        transcript = " ".join(transcripts).strip()
        print(f"[{lesson_id}] Trascrizione completa assemblata.")
        
        # --- 4 & 5. RIASSUNTO E ARGOMENTO (UNIFICATI IN UNA CHIAMATA) ---
        unified_prompt = f"""Sei un assistente IA esperto nella redazione di testi accademici per la materia di {subject}. Il tuo compito è trasformare la trascrizione di una lezione in una sintesi didattica che sia al contempo **esaustiva nel contenuto e impeccabile nella struttura**. Devi eseguire due compiti e restituire il risultato come un singolo oggetto JSON.

**REGOLA FONDAMENTALE: Massima Fedeltà alla Trascrizione**
La tua intera elaborazione deve basarsi **ESCLUSIVAMENTE** sul contenuto della trascrizione fornita. **NON DEVI** usare conoscenza esterna. L'obiettivo è valorizzare e strutturare il materiale esistente al suo massimo potenziale, non integrarlo con informazioni nuove.

**COMPITI:**
1.  **Creare una Sintesi Esaustiva e Fedele:**
    *   **Obiettivo:** Trasformare il contenuto della trascrizione in un testo scritto che sia completo, dettagliato e perfettamente organizzato, riflettendo fedelmente la profondità della lezione originale.

    **LINEE GUIDA OBBLIGATORIE PER IL CONTENUTO E LO STILE:**
    *   **Completezza Assoluta (Priorità #1):** Il tuo primo obiettivo è la completezza. Assicurati di includere **TUTTI** i concetti, le definizioni, gli esempi, le analogie e le spiegazioni presenti nella trascrizione. **Non operare semplificazioni o omissioni per brevità**. La sintesi deve essere un riflesso ricco e dettagliato del contenuto della lezione. Ogni informazione rilevante deve essere catturata.
    *   **Riorganizzazione Logica (Priorità #2):** Una volta catturati tutti i contenuti, il tuo secondo compito è organizzarli in una struttura gerarchica chiara (titoli, sottotitoli, elenchi puntati). Trasforma il flusso spesso non lineare del parlato in un percorso di apprendimento chiaro e sequenziale.
    *   **Descrizione Dettagliata dei Processi:** Quando la trascrizione descrive un processo o un meccanismo, ricostruiscine le fasi con il **massimo livello di dettaglio consentito dal testo**. Elenca tutti gli attori (molecole, enzimi, ecc.) menzionati e il loro ruolo, così come descritto.
    *   **Connessioni Logiche Esplicite:** Identifica e rendi esplicite le relazioni di causa-effetto e i collegamenti tra argomenti diversi menzionati nella lezione, anche se non sono immediatamente consecutivi nel parlato.
    *   **Riformulazione Accademica:** Mantieni uno stile formale e preciso, eliminando ripetizioni e colloquialisms tipici del discorso orale, ma senza perdere il contenuto informativo.

    **FORMATTAZIONE OBBLIGATORIA DEL RIASSUNTO:**
    *   Usa la sintassi Markdown standard (es. `## Titolo`, `### Sottotitolo`, `* elenco puntato`, `**grassetto**`).
    *   Utilizza la sintassi LaTeX per **tutte** le formule, espressioni, simboli matematici e cariche ioniche menzionate nella trascrizione (es. $f(x)=x^2$, $Na^+$, $Cl^-$).

2.  **Generare un Titolo (Argomento) per la Lezione:**
    *   Il titolo deve essere conciso, accademico e descrittivo, composto da un massimo di 5-7 parole, riflettendo l'intero contenuto della trascrizione.

**ISTRUZIONI DI OUTPUT:**
- Restituisci **ESCLUSIVAMENTE** un oggetto JSON valido. Non includere testo introduttivo, spiegazioni o ```json...```.
- L'oggetto JSON deve avere due chiavi: `summary` (stringa) e `suggestedTopic` (stringa).

**Trascrizione della lezione:**
---
{transcript}
---
"""
        unified_response = None
        max_retries = 3
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        for attempt in range(max_retries):
            try:
                print(f"[{lesson_id}] Generazione riassunto e argomento (Tentativo {attempt + 1})...")
                unified_response = summary_model.generate_content(
                    unified_prompt, 
                    generation_config=generation_config,
                    request_options={'timeout': 1800}
                )
                break
            except exceptions.ResourceExhausted as e:
                wait_time = (2 ** attempt) * 5 + random.uniform(0, 1)
                print(f"[{lesson_id}] Rate limit superato durante il riassunto/argomento. Dettagli API: {e}. Attendo {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        if unified_response is None:
            raise RuntimeError("Impossibile generare riassunto e argomento.")

        try:
            cleaned_text = unified_response.text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            
            result_json = json.loads(cleaned_text)
            summary = result_json['summary']
            suggested_topic = result_json['suggestedTopic'].strip().replace('"', '')
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[{lesson_id}] ERRORE: impossibile decodificare il JSON per riassunto/argomento. Risposta ricevuta: {unified_response.text}")
            raise RuntimeError(f"Decodifica JSON fallita: {e}")

        job_data = { "status": "completed", "result": { "transcript": transcript, "summary": summary, "suggestedTopic": suggested_topic } }
        with open(job_file, 'w') as f:
            json.dump(job_data, f)
        print(f"[{lesson_id}] Elaborazione completata con successo.")

    except exceptions.ResourceExhausted as e:
        print(f"ERRORE GRAVE (QUOTA ESAURITA) durante l'elaborazione per [{lesson_id}]: {e}")
        error_data = {"status": "error", "message": f"RATE_LIMIT_EXCEEDED::{e}"}
        with open(job_file, 'w') as f:
            json.dump(error_data, f)
    except Exception as e:
        print(f"ERRORE GRAVE durante l'elaborazione per [{lesson_id}]: {e}")
        error_data = {"status": "error", "message": str(e)}
        with open(job_file, 'w') as f:
            json.dump(error_data, f)
    finally:
        if sanitized_temp_file and sanitized_temp_file.exists():
            sanitized_temp_file.unlink()
            print(f"[{lesson_id}] File temporaneo sanitizzato {sanitized_temp_file} eliminato.")
        if raw_input_file and raw_input_file.exists():
            raw_input_file.unlink()
            print(f"[{lesson_id}] File temporaneo di input {raw_input_file} eliminato.")


# --- ENDPOINT DELL'API ---
# NESSUNA MODIFICA: Tutti gli endpoint e la loro logica sono invariati.
@app.before_request
def check_api_key():
    if request.method == 'OPTIONS': return
    if request.endpoint not in ['status', 'run_app']:
        if not request.headers.get('X-API-Key'):
            return jsonify({"error": "Header X-API-Key mancante"}), 401

@app.route('/status', methods=['GET'])
def status(): 
    return jsonify({"status": "ok"})

@app.route('/upload', methods=['POST'])
def upload_file():
    api_key = request.headers.get('X-API-Key')
    user_hash = get_user_hash(api_key)
    user_dir = JOBS_DIR / user_hash
    user_dir.mkdir(exist_ok=True)
    if 'file' not in request.files: return jsonify({"error": "Nessun file fornito"}), 400
    
    file = request.files['file']
    subject = request.form.get('subject', 'N/A')
    transcription_model = request.form.get('transcription_model', 'gemini-1.5-flash')
    summary_model = request.form.get('summary_model', 'gemini-1.5-pro')
    
    suffix = Path(file.filename).suffix if file.filename else '.tmp'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_f:
        raw_input_path = Path(temp_f.name)
        file.save(raw_input_path)

    lesson_id = f"lesson_{int(time.time() * 1000)}"
    with open(user_dir / f"{lesson_id}.json", 'w') as f: json.dump({"status": "processing"}, f)
    
    thread = threading.Thread(target=transcribe_and_summarize_task, args=(lesson_id, user_hash, api_key, str(raw_input_path), subject, transcription_model, summary_model))
    thread.start()
    
    print(f"Nuovo job creato: utente={user_hash[:8]}, ID={lesson_id}, Trascrizione={transcription_model}, Riassunto={summary_model}")
    return jsonify({"lesson_id": lesson_id})

@app.route('/result/<lesson_id>', methods=['GET'])
def get_result(lesson_id):
    api_key = request.headers.get('X-API-Key')
    user_hash = get_user_hash(api_key)
    job_file = JOBS_DIR / user_hash / f"{lesson_id}.json"
    
    if not job_file.exists(): return jsonify({"status": "not_found"}), 404
    
    with open(job_file, 'r') as f: return jsonify(json.load(f))

@app.route('/sync', methods=['POST'])
def sync_results():
    api_key = request.headers.get('X-API-Key')
    user_hash = get_user_hash(api_key)
    user_dir = JOBS_DIR / user_hash

    if not user_dir.exists(): return jsonify([])

    ids_to_check = request.json.get('known_ids', [])
    completed_jobs = []
    
    files_to_check = [user_dir / f"{lid}.json" for lid in ids_to_check] if ids_to_check else user_dir.glob("*.json")

    for job_file in files_to_check:
        if not job_file.exists(): continue
        with open(job_file, 'r') as f:
            try:
                data = json.load(f)
                if data.get("status") in ["completed", "error"]:
                    completed_jobs.append({"lesson_id": job_file.stem, **data})
            except json.JSONDecodeError:
                print(f"Attenzione: file job corrotto durante sync: {job_file}")

    if len(completed_jobs) > 0:
        print(f"Sincronizzazione per {user_hash[:8]}: trovati {len(completed_jobs)} nuovi risultati.")
        
    return jsonify(completed_jobs)

# --- BLOCCO DI AVVIO SERVER ---
# <-- RIMOSSO: La funzione run_app() e il blocco if __name__ == '__main__' sono stati eliminati.
# Un server di produzione come Gunicorn avvierà l'applicazione direttamente,
# usando l'oggetto 'app' definito in questo file.
