import streamlit as st
import google.generativeai as genai
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.generativeai.types import content_types

# Set page config
st.set_page_config(page_title="AI Productivity Assistant", page_icon="🤖", layout="wide")

# --- INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "notes" not in st.session_state:
    st.session_state.notes = []
if "finance" not in st.session_state:
    st.session_state.finance = []
if "schedule" not in st.session_state:
    st.session_state.schedule = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat" not in st.session_state:
    st.session_state.chat = None
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False

# --- AGENT TOOLS (FUNCTIONS) ---
def add_note(topic: str, content: str) -> str:
    """Simpan catatan kuliah, rapat, atau informasi penting lainnya. Parameters: topic (topik), content (isi catatan)."""
    st.session_state.notes.append({"topic": topic, "content": content})
    return f"Catatan tentang '{topic}' berhasil disimpan di memori sistem."

def get_notes() -> str:
    """Ambil semua catatan yang pernah disimpan sebelumnya di memori sistem."""
    if not st.session_state.notes:
        return "Belum ada catatan yang disimpan."
    result = "Catatan Anda:\n"
    for i, note in enumerate(st.session_state.notes):
        result += f"{i+1}. [{note['topic']}] {note['content']}\n"
    return result

def add_finance(tipe: str, amount: int, description: str) -> str:
    """Catat keuangan pribadi. tipe HARUS diisi dengan 'pemasukan' atau 'pengeluaran'."""
    if tipe.lower() not in ['pemasukan', 'pengeluaran']:
        return "Gagal: Tipe harus 'pemasukan' atau 'pengeluaran'."
    st.session_state.finance.append({"type": tipe.lower(), "amount": amount, "description": description})
    return f"{tipe.capitalize()} sebesar {amount} berhasil dicatat dengan deskripsi: {description}."

def get_finance_summary() -> str:
    """Dapatkan ringkasan laporan keuangan (total pemasukan, total pengeluaran, saldo akhir)."""
    if not st.session_state.finance:
        return "Belum ada catatan keuangan."
    pemasukan = sum([item['amount'] for item in st.session_state.finance if item['type'] == 'pemasukan'])
    pengeluaran = sum([item['amount'] for item in st.session_state.finance if item['type'] == 'pengeluaran'])
    saldo = pemasukan - pengeluaran
    
    rincian = "\n".join([f"- {item['type'].capitalize()}: {item['amount']} ({item['description']})" for item in st.session_state.finance])
    
    return f"Ringkasan Keuangan:\nTotal Pemasukan: {pemasukan}\nTotal Pengeluaran: {pengeluaran}\nSaldo Saat Ini: {saldo}\n\nRincian:\n{rincian}"

def add_schedule(task: str, date_time: str) -> str:
    """Tambahkan jadwal atau pengingat baru. Parameters: task (nama tugas/acara), date_time (waktu, misal 'Besok jam 10 pagi')."""
    st.session_state.schedule.append({"task": task, "time": date_time})
    return f"Jadwal '{task}' pada '{date_time}' berhasil ditambahkan ke kalender internal."

def get_schedule() -> str:
    """Lihat semua jadwal dan pengingat yang ada di kalender internal."""
    if not st.session_state.schedule:
        return "Jadwal kosong. Belum ada pengingat."
    result = "Jadwal Anda:\n"
    for i, sch in enumerate(st.session_state.schedule):
        result += f"{i+1}. {sch['task']} (Waktu: {sch['time']})\n"
    return result

def search_document(query: str) -> str:
    """Gunakan ini HANYA jika Anda ditanya tentang isi dokumen/PDF yang diunggah pengguna. Mencari informasi di dokumen eksternal menggunakan RAG."""
    if st.session_state.vector_store is None:
        return "Gagal mencari: Belum ada dokumen PDF yang diunggah oleh pengguna ke dalam sistem RAG."
    docs = st.session_state.vector_store.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    return f"Konteks dari dokumen yang relevan ditemukan:\n{context}\n\nJawablah pertanyaan berdasarkan konteks ini."

# List of tools to pass to Gemini
tools_list = [add_note, get_notes, add_finance, get_finance_summary, add_schedule, get_schedule, search_document]

# --- SIDEBAR & CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    api_key = st.text_input("Gemini API Key", type="password")
    
    tone = st.selectbox(
        "Gaya Bahasa (Tone)",
        ["Profesional & Formal", "Santai & Ramah (Bergaul)"]
    )
    
    st.divider()
    st.header("📄 Unggah Dokumen (RAG)")
    st.markdown("Unggah PDF untuk mengekstrak informasi dengannya.")
    uploaded_file = st.file_uploader("Upload file PDF", type=["pdf"])
    
    if st.button("Proses Dokumen"):
        if not api_key:
            st.error("Silakan masukkan API Key terlebih dahulu.")
        elif uploaded_file is not None:
            with st.spinner("Memproses dokumen dengan RAG..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # RAG Pipeline (Load -> Chunk -> Embed -> Store)
                    loader = PyPDFLoader(tmp_path)
                    pages = loader.load_and_split()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(pages)
                    
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    st.session_state.vector_store = vector_store
                    
                    st.success("Dokumen berhasil diunggah dan diindeks! Anda sekarang bisa bertanya tentang isi dokumen ini.")
                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses PDF: {e}")
        else:
            st.warning("Pilih file PDF terlebih dahulu.")

# --- MAIN APP UI ---
st.title("🚀 Personal Productivity Assistant")
st.markdown("""
Asisten AI cerdas berbasis Gemini yang dapat membantu Anda:
1. 🗓️ Mengelola Jadwal & Pengingat
2. 📝 Membuat Catatan Kuliah/Rapat
3. 💰 Mencatat & Melaporkan Keuangan Pribadi
4. 📄 Menjawab Pertanyaan dari Dokumen PDF (RAG)
""")

# Initialize or re-initialize Chat model if settings change
if api_key:
    genai.configure(api_key=api_key)
    # Define System Instruction
    sys_instruct = f"""Anda adalah Personal Productivity Assistant yang cerdas. 
Tugas Anda adalah membantu user mengatur jadwal, mencatat uang, menyimpan catatan rapat/kuliah, dan membaca dokumen.
Gaya Bahasa yang harus Anda gunakan: {tone}.
Anda MENDUKUNG function calling (tools). Gunakan alat yang tersedia jika pengguna meminta Anda untuk mencatat, melihat jadwal, melihat keuangan, atau mencari informasi dari dokumen.
Jika pengguna menanyakan sesuatu tentang isi dokumen, selalu gunakan fungsi search_document."""

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            tools=tools_list,
            system_instruction=sys_instruct
        )
        
        # We need to maintain the chat history specifically for function calling
        if st.session_state.chat is None or st.session_state.api_key_valid == False:
            st.session_state.chat = model.start_chat(enable_automatic_function_calling=True)
            st.session_state.api_key_valid = True
            
    except Exception as e:
        st.error(f"Gagal inisialisasi model: {e}")
        st.session_state.api_key_valid = False
else:
    st.info("👈 Masukkan Gemini API Key Anda di sidebar untuk memulai.")
    st.session_state.api_key_valid = False

st.divider()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Apa yang bisa saya bantu hari ini?"):
    if not st.session_state.api_key_valid:
        st.error("API Key belum diisi atau tidak valid!")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response using Agent Chat
        with st.chat_message("assistant"):
            with st.spinner("Berpikir dan mengeksekusi..."):
                try:
                    # Send message to Gemini Chat (automatic function calling is enabled)
                    response = st.session_state.chat.send_message(prompt)
                    
                    # Display response
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menghubungi API: {e}")
