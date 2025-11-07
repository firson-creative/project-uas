import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Memuat model 'all-MiniLM-L6-v2'...")
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("...Model dimuat.")

print("Memuat database vektor dan chunks...")
try:
    DATABASE_VEKTOR = np.load("vectors.npy")
    with open("chunks.json", "r", encoding="utf-8") as f:
        CHUNKS = json.load(f)

    print(f"...Berhasil membuat {len(CHUNKS)} potongan teks.")

except FileNotFoundError:
    print("ERROR: File 'vectors.npy' atau 'chunks.json' tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan 'ekstraksi.py' telebih dahulu.")
    exit()

def cari_jawaban(pertanyaan_user):
    vektor_pertanyaan = MODEL.encode(pertanyaan_user)
    skor_kemiripan = cosine_similarity(
        vektor_pertanyaan.reshape(1, -1),
        DATABASE_VEKTOR
    )

    skor = skor_kemiripan[0]
    indeks_terbaik = np.argmax(skor)
    skor_terbaik = skor[indeks_terbaik]
    jawaban_terbaik = CHUNKS[indeks_terbaik]

    return jawaban_terbaik, float(skor_terbaik)

if __name__ == '__main__':
    print("\n--- Sistem Tanya Jawab PDF ---")
    print("Ketik 'keluar untuk berhenti.")

    while True:
        pertanyaan = input("\n[Kamu] > ")
        if pertanyaan.lower() == 'keluar':
            print("[ASK-PDF] > Sampai Jumpa!")
            break

        jawaban, skor = cari_jawaban(pertanyaan)

        print("\n[ASK-PDF] > Jawaban paling relevean ditemukan (Skor: {:.4f}:)".format(skor))
        print("------------------------------------------")
        print(jawaban)
        print("------------------------------------------")