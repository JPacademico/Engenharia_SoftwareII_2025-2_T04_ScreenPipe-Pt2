import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

assert torch.cuda.is_available()

DEVICE = "cuda"

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
GEN_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

def carregar_texto(caminho):
    with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def gerar_chunks(texto, chunk_size=1200):
    return [
        texto[i:i + chunk_size].strip()
        for i in range(0, len(texto), chunk_size)
        if texto[i:i + chunk_size].strip()
    ]

def gerar_conclusao(textos_a, textos_b):
    prompt = f"""<s>[INST]
Você recebeu dois conjuntos de textos técnicos relacionados a engenharia de software.
Elabore uma conclusão longa, contínua e técnica, em múltiplos parágrafos corridos, sem listas.

A conclusão deve:
- Identificar o tema central comum
- Explicar detalhadamente as convergências conceituais
- Analisar diferenças de intenção, profundidade e público-alvo
- Avaliar se um texto deriva conceitualmente do outro
- Explicar o papel de cada texto no contexto de análise versus documentação
- Encerrar com uma avaliação consolidada do grau de alinhamento semântico

Texto A:
{' '.join(textos_a)}

Texto B:
{' '.join(textos_b)}

Conclusão técnica consolidada:
[/INST]
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(gen_model.device)

    output = gen_model.generate(
        **inputs,
        max_new_tokens=900,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )

    texto = tokenizer.decode(output[0], skip_special_tokens=True)
    return texto.split("Conclusão técnica consolidada:")[-1].strip()

def comparar_arquivos(arquivo_a, arquivo_b, top_k=6):
    texto_a = carregar_texto(arquivo_a)
    texto_b = carregar_texto(arquivo_b)

    chunks_a = gerar_chunks(texto_a)
    chunks_b = gerar_chunks(texto_b)

    emb_a = embed_model.encode(
        chunks_a,
        batch_size=64,
        normalize_embeddings=True
    )

    emb_b = embed_model.encode(
        chunks_b,
        batch_size=64,
        normalize_embeddings=True
    )

    matriz = cos_sim(emb_a, emb_b).cpu().numpy()

    similares = []
    for i in np.argsort(matriz.max(axis=1))[::-1][:top_k]:
        j = matriz[i].argmax()
        similares.append(
            (chunks_a[i], chunks_b[j], float(matriz[i, j]))
        )

    conclusao_textual = gerar_conclusao(
        [s[0][:900] for s in similares],
        [s[1][:900] for s in similares]
    )

    return {
        "conclusao_textual": conclusao_textual,
        "pares_relevantes": [
            {
                "score": s[2],
                "trecho_a": s[0][:500],
                "trecho_b": s[1][:500]
            } for s in similares
        ]
    }

ARQUIVO_1 = "./arquivo_a.txt"
ARQUIVO_2 = "./arquivo_b.txt"

resultado = comparar_arquivos(
    ARQUIVO_1,
    ARQUIVO_2
)

print("\n=== CONCLUSÃO TÉCNICA CONSOLIDADA ===\n")
print(resultado["conclusao_textual"])

print("\n=== PRINCIPAIS PARES SEMELHANTES ===\n")
for i, p in enumerate(resultado["pares_relevantes"], 1):
    print(f"\n--- PAR {i} | Score: {p['score']:.4f} ---")
    print("\n[Arquivo A]")
    print(p["trecho_a"])
    print("\n[Arquivo B]")
    print(p["trecho_b"])
