import json
import os
import fitz
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

PDF_FOLDER_PATH = "data_pdf"

def ekstrak_teks_df(folder_path):
    print(f"Memindai folder: {folder_path}...")
    
    teks_lengkap = ""

    for nama_file in os.listdir(folder_path):

        if nama_file.lower().endswith(".pdf"):
            print(f"  Membaca file (via PyMuPDF): {nama_file}...")
            
            path_file_lengkap = os.path.join(folder_path, nama_file)
            
            try:
                with fitz.open(path_file_lengkap) as doc:

                    for halaman in doc:
                        teks_halaman = halaman.get_text().strip()
                        
                        if teks_halaman:
                            teks_lengkap += teks_halaman + "\n"
                            
            except Exception as e:
                print(f"    ERROR: Tidak bisa membaca {nama_file}. Error: {e}")
                
    print("...Ekstraksi selesai.")
    return teks_lengkap


def chunk_teks(teks, chunk_size=1000, overlap=200):

    print(f"\nMemulai chunking teks...")
    print(f"Ukuran Chunk: {chunk_size} karakter, Overlap: {overlap} karakter.")
    chunks = []

    panjang_teks = len(teks)
    indeks_awal = 0

    while indeks_awal < panjang_teks:
        indeks_akhir = indeks_awal + chunk_size
        potongan = teks[indeks_awal:indeks_akhir]
        potongan_bersih = potongan.strip()
        if potongan_bersih:
            chunks.append(potongan_bersih)
            
        indeks_awal += chunk_size - overlap

    print(f"...Chunking selesai. Total potongan dibuat: {len(chunks)}")
    return chunks

def db_vektor(chunks): #memebuat database vektor

    print(f"\nMemulai Modelling Vektorisasi (Embedding)...")
    print(" Memuat model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f" Mengubah {len(chunks)} potongan teks menjadi vektor...")

    embeddings= model.encode(chunks, show_progress_bar=True)
    print("...VEktorisasi selesai.")
    return embeddings
    
    
if __name__ == "__main__":
    
    semua_teks = ekstrak_teks_df(PDF_FOLDER_PATH)
    potongan_teks = chunk_teks(semua_teks)
    database_vektor = db_vektor(potongan_teks)

    print("\nMenyinmpan hasil ke file...")

    try:
        with open("chunks.json", "w", encoding="utf-8") as f:
            json.dump(potongan_teks, f, indent=2)

        np.save("vectors.npy", database_vektor)

        print("...Hasil berhasil disimpan sebagai 'chunks.json' dan 'vectors.npy'")

    except Exception as e:
        print(f"Error saat menyimpan file: {e}")

    print("-----------------------------------------")
    print(f"Total potongan (chunks) dibuat: {len(potongan_teks)}")

    if 'database_vektor' in locals():
        print(f"Dimensi Database Vektor: {database_vektor.shape}")