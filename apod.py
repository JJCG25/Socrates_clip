import pandas as pd
import asyncio
import aiohttp
import os
import re
from PIL import Image
from io import BytesIO

# --- CONFIGURACIÓN ---
INPUT_JSON = 'tu_archivo_apod.json'
OUTPUT_FOLDER = 'apod_images_224'
OUTPUT_CSV = 'apod_finetuning_dataset.csv'

def clean_text(text):
    """Limpieza básica para eliminar ruido antes del LLM o entrenamiento"""
    if not isinstance(text, str): return ""
    text = re.split(r'Copyright|Image Credit|Credit:', text, flags=re.IGNORECASE)[0]
    text = re.sub(r'http\S+', '', text)
    text = text.replace('\n', ' ').strip()
    return text

async def download_and_preprocess(session, row, folder):
    url = row['url']
    filename = os.path.join(folder, f"{row['date']}.jpg")
    if os.path.exists(filename): return True # Ya descargada

    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                content = await response.read()
                img = Image.open(BytesIO(content)).convert('RGB')
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                img.save(filename, "JPEG", quality=90)
                return True
    except:
        return False
    return False

async def main_pipeline():
    df = pd.read_json(INPUT_JSON)
    df = df[df['media_type'] == 'image'].copy()
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print(f"Iniciando descarga de {len(df)} imágenes...")
    
    conn = aiohttp.TCPConnector(limit_per_host=10)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [download_and_preprocess(session, row, OUTPUT_FOLDER) for _, row in df.iterrows()]
        results = await asyncio.gather(*tasks)

    df['downloaded'] = results
    df = df[df['downloaded'] == True] 
    
    df['clean_caption'] = df['explanation'].apply(clean_text)
    df['clean_caption'] = df['clean_caption'].str.slice(0, 300) 
    
    # Guardar CSV final para el entrenamiento
    dataset_final = df[['date', 'clean_caption']]
    dataset_final['image_path'] = dataset_final['date'].apply(lambda x: os.path.join(OUTPUT_FOLDER, f"{x}.jpg"))
    
    dataset_final.to_csv(OUTPUT_CSV, index=False)
    print(f"¡Listo! Dataset guardado en {OUTPUT_CSV}")

# Ejecutar
asyncio.run(main_pipeline())