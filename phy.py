import cohere
import pandas as pd
from tqdm import tqdm
import time

COHERE_API_KEY = 'cohere_DGuUVmLmgbBO3b0E8rqWusMYfgWFkKghR6x6TFFZ4NCfd3' # Tu key de $500 USD
INPUT_FILE = "dataset_clip_ready.csv"
OUTPUT_FILE = "dataset_clip_philosophical.csv"

co = cohere.Client(COHERE_API_KEY)
df = pd.read_csv(INPUT_FILE)

df['input_text'] = (df['title'].fillna('') + " " + df['caption'].fillna('')).str.strip()

vacias = df[df['input_text'] == ''].shape[0]
print(f"Eliminando {vacias} registros sin información textual.")

df = df[df['input_text'] != ''].reset_index(drop=True)

df = df.dropna(subset=['image_path', 'input_text']).reset_index(drop=True)

PREAMBLE = """You are an existentialist philosopher, expert in the relationship between the cosmic void and the human condition. 
Your style is sober, poetic, and deeply melancholic, akin to the prose of Albert Camus and the brevity of Emil Cioran. 
Your task is to transmute cold astronomical data into piercing existential truths. 
ALL RESPONSES MUST BE IN ENGLISH."""

def get_prompt(batch):
    batch_str = "\n".join([f"{i+1}. {t}" for i, t in enumerate(batch)])
    return f"""Transmute these {len(batch)} DISTINCT NASA observations into UNIQUE existentialist aphorisms.

    STRICT PROTOCOL:
    1. NO REPETITION: Every aphorism must be unique and specifically tailored to the image description.
    2. NO GENERIC PHRASES: Avoid overusing 'void', 'fleeting sparks', or 'darkness' unless strictly necessary.
    3. STRUCTURE: Provide exactly one aphorism per input, separated by '|||'.
    4. LANGUAGE: English.

    Input Descriptions:
    {batch_str}

    Aphorisms (Separated by '|||'):"""

batch_size = 5  
all_results = []

print(f"Iniciando Transmutación Filosófica de {len(df)} registros...")

batch_size = 5
all_results = []

for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df['input_text'].iloc[i:i+batch_size].tolist()
    
    try:
        response = co.chat(
            model='command-r7b-12-2024',
            message=get_prompt(batch_texts),
            preamble=PREAMBLE,
            temperature=1, 
            presence_penalty=0.8
        )
        

        raw_text = response.text
        
        lines = [line.strip() for line in raw_text.split('NEXT') if len(line.strip()) > 5]
        
        if len(lines) < len(batch_texts):
            lines = [line.strip() for line in raw_text.split('\n') if len(line.strip()) > 10]

        while len(lines) < len(batch_texts):
            lines.append("The universe offers no witness today.")
        
        batch_results = lines[:len(batch_texts)]
        all_results.extend(batch_results)
        
        # GUARDADO INMEDIATO: No esperamos a terminar
        current_df = df.iloc[:len(all_results)].copy()
        current_df['phil_caption'] = all_results
        current_df.to_csv(OUTPUT_FILE, index=False)

    except Exception as e:
        print(f"Error en batch {i}: {e}")
        all_results.extend(["Error."] * len(batch_texts))
"""
for i in tqdm(range(0, 2, batch_size)):
    batch_texts = df['input_text'].iloc[i:i+batch_size].tolist()
    
    try:
        response = co.chat(
            model='command-r7b-12-2024',
            message=get_prompt(batch_texts),
            preamble=PREAMBLE,
            temperature=1,
            presence_penalty=0.8
        )

        batch_results = [r.strip() for r in response.text.split('|||') if r.strip()]
        
        while len(batch_results) < len(batch_texts):
            batch_results.append("The universe didn't bother to answer.")
            
        all_results.extend(batch_results[:len(batch_texts)])
        
    except Exception as e:
        print(f"\nError en el batch {i}: {e}")
        
        all_results.extend(["Error en la transmutación cósmica."] * len(batch_texts))
        time.sleep(1)


df['phil_caption'] = all_results[2]
df.to_csv(OUTPUT_FILE, index=False)

print(f"\n¡Listo! El dataset está completo.")
print(f"Archivo guardado como: {OUTPUT_FILE}")
"""