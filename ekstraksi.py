import os
from PyPDF2 import PdfReader

PDF_FOLDER_PATH = "data_pdf"

def ekstrak_teks_df(folder_path):

    print(f"Memindai foder: {folder_path}...")

    teks_lengkap = ""

    for nama_file in os.listdir(folder_path):
        if nama_file.endswith(".pdf"):
            print(f" membaca file: {nama_file}...")
            path_file_lengkap = os.path. join(folder_path, nama_file)

            try:
                with open(path_file_lengkap, 'rb') as f:
                    reader = PdfReader(f)

                    for halaman in reader.pages:
                        teks_halaman = halaman.extract_text()
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
        potongan = teks[indeks_awal + chunk_size]
        chunks.append(potongan)

        indeks_awal += chunk_size - overlap

    print(f"...Chunking selesai. Total potongan dibuat: {len(chunks)}")
    return chunks
    
if __name__ == "__main__":
    
    semua_teks = ekstrak_teks_df(PDF_FOLDER_PATH)
    potongan_teks = chunk_teks(semua_teks)
    print("\n--- HASIL CHUNCKING ---")

    if potongan_teks:
        print(potongan_teks[0])
        print("...")
        print(f"\nTotal potongan (chunks) dibuat: {len(potongan_teks)}")
    else:
        print("Tidak ada teks yang berhasil di-chunk.")

    print("-----------------------------------------")