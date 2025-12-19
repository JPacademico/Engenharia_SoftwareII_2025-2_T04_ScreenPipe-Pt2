# Documentação Técnica Completa

## Pipeline de Análise Semântica, Releases e Comparação Textual com LLMs

------------------------------------------------------------------------

## Sumário

1.  Visão Geral
2.  Objetivos do Sistema
3.  Modelos Utilizados e Motivação
4.  Arquitetura Geral
5.  Dependências e Requisitos\
6.  Como Executar o Estudo (Instruções de Reprodutibilidade)
7.  Pipeline de Análise de Releases
8.  Pipeline Alternativo com Modelo Leve
9.  Pipeline de Comparação Semântica
10.  Geração de Conclusão Técnica
11.  Estrutura dos Resultados
12. Boas Práticas Adotadas
13. Possíveis Extensões
14. Conclusão Final

------------------------------------------------------------------------

## 1. Visão Geral

Este projeto implementa um pipeline avançado de análise semântica
aplicada à engenharia de software, combinando Modelos de Linguagem
de Grande Porte (LLMs) com modelos de embeddings vetoriais para
análise técnica profunda de artefatos textuais.

O sistema foi projetado para execução local com **aceleração por GPU**,
utilizando quantização em 4 bits, permitindo alto desempenho com uso
eficiente de memória.

------------------------------------------------------------------------

## 2. Objetivos do Sistema

O pipeline atende a três objetivos centrais:

-   Analisar históricos de releases para inferir práticas de
    versionamento e fluxo de trabalho.
-   Comparar documentos técnicos para identificar convergência semântica
    e possível derivação conceitual.
-   Produzir conclusões técnicas discursivas, adequadas para
    documentação, auditoria ou análise arquitetural.
    
------------------------------------------------------------------------

## 3. Modelos Utilizados e Motivação

### BAAI/bge-base-en-v1.5 (BGE-Base)

**Motivação:**  
Utilizado como modelo central de embeddings semânticos em todo o pipeline de comparação textual. O **bge-base-en-v1.5** foi escolhido por apresentar excelente equilíbrio entre qualidade semântica, velocidade de inferência e baixo consumo de memória. No sistema, ele é responsável pela vetorização de fragmentos textuais normalizados, permitindo o cálculo eficiente de similaridade via cosseno entre documentos técnicos. Sua adoção é fundamental para a identificação objetiva de convergências conceituais antes da etapa de inferência com LLMs, reduzindo ruído e delimitando com precisão os trechos mais relevantes para análise aprofundada.

### Qwen 2.5 – 1.5B Instruct

**Motivação:**  
Empregado como modelo leve de apoio nas fases iniciais de triagem e contextualização. A escolha do Qwen 2.5 (1.5B parâmetros) decorre de sua capacidade de gerar interpretações coerentes com baixo custo computacional, sendo adequado para ambientes com restrição de recursos. No contexto do projeto, ele atua como mecanismo de análise preliminar e validação cruzada, possibilitando iterações rápidas sem sobrecarregar a GPU.

### Mistral-7B-Instruct-v0.3

**Motivação:**  
Responsável pela síntese técnica e geração das conclusões discursivas consolidadas. O Mistral-7B foi selecionado por seu forte desempenho em tarefas de raciocínio técnico e produção de texto estruturado. Para viabilizar sua execução em GPUs intermediárias (RTX 4060 ou Tesla T4), o modelo é utilizado com quantização em 4 bits, mantendo qualidade de inferência sem exceder os limites de VRAM. Ele opera exclusivamente sobre os trechos previamente selecionados pelo pipeline de similaridade, garantindo foco analítico e maior auditabilidade.

### Meta-LLaMA 3.1 – 8B Instruct

**Motivação:**  
Utilizado na etapa final de avaliação crítica e validação conceitual das conclusões obtidas. O LLaMA 3.1–8B foi escolhido por seu elevado grau de capacidade lógica e consistência analítica, sendo empregado como um validador técnico das interpretações relacionadas a práticas de engenharia de software, workflows e estratégias de versionamento. Seu papel é confirmar aderência a modelos consolidados, como GitHub Flow e Rapid Releases, reforçando a robustez metodológica do pipeline.

------------------------------------------------------------------------

## 4. Arquitetura Geral

A arquitetura é organizada em três camadas lógicas, com
responsabilidades bem definidas.

### 4.1 Camada de Inferência com LLMs

Responsável pela interpretação semântica profunda, geração de análises
técnicas e síntese discursiva.

Modelos utilizados: - **Mistral-7B-Instruct v0.3** - **Qwen 2.5 1.5B
Instruct**

### 4.2 Camada de Similaridade Semântica

Responsável pela vetorização e comparação matemática entre textos.

Modelo utilizado: - **BAAI/bge-base-en-v1.5**

### 4.3 Camada de Orquestração

Responsável por leitura de arquivos, fragmentação textual, controle de
memória, construção de prompts e consolidação de resultados.

------------------------------------------------------------------------

## 5. Dependências e Requisitos

### 5.1 Bibliotecas

-   Clone o repositório: git clone [URL
-   Instale as bibliotecas: pip install -r requirements.txt
-   Execute o script principal: python src/main.py (ou abra os notebooks em src/modelos_hugging_face/).

### 5.2 Requisitos de Hardware

-   GPU: NVIDIA GeForce RTX 4060 (8GB VRAM) / Tesla T4 (Colab).
-   Processador: AMD Ryzen 7 5800X 8-Core 3.80GHz
-   Memória RAM: 32GB DDR4.

------------------------------------------------------------------------

## 6. Como Executar o Estudo (Instruções de Reprodutibilidade)

Este guia permite a replicação integral dos resultados apresentados no tutorial.

Clone o Repositório:

```bash
git clone https://github.com/JPacademico/Engenharia_SoftwareII_2025-2_T04_ScreenPipe-Pt2.git
cd Engenharia_SoftwareII_2025-2_T04_ScreenPipe-Pt2
```

Instale as Dependências:

```bash
pip install -r requirements.txt
```

Execução do Pipeline por Etapas: O projeto está dividido em notebooks específicos para cada fase da análise, localizados em src/modelos_hugging_face/:

-   Etapa 1 -> Análise Estrutural: Execute o Qwen.ipynb para a triagem inicial.
-   Etapa 2 -> Síntese Técnica: Execute o Mistral.ipynb para identificação de padrões.
-   Etapa 3 -> Avaliação Crítica: Execute o llama.ipynb para a validação final das estratégias de engenharia.

------------------------------------------------------------------------

## 6. Pipeline de Análise de Releases

Este pipeline processa logs de releases para identificar:

-   Estratégia de versionamento
-   Cadência de entregas
-   Modelo de workflow (GitFlow, Trunk-based, híbrido)

Utiliza prompts estruturados no formato nativo do modelo, com geração
determinística e controle rigoroso de contexto.

------------------------------------------------------------------------

## 7. Pipeline Alternativo com Modelo Leve

O modelo **Qwen 1.5B** é utilizado como alternativa ultraleve, indicado
para:

-   Ambientes com restrição de VRAM
-   Execuções rápidas
-   Validação cruzada de análises

Inclui limpeza explícita de memória e execução com `torch.no_grad()`.

------------------------------------------------------------------------

## 8. Pipeline de Comparação Semântica

### 8.1 Fragmentação

Os documentos são divididos em blocos semanticamente coerentes, evitando
truncamento excessivo.

### 8.2 Vetorização

Cada fragmento é convertido em embedding vetorial normalizado.

### 8.3 Similaridade

A similaridade é calculada via **cosseno**, selecionando os pares mais
relevantes.

------------------------------------------------------------------------

## 9. Geração de Conclusão Técnica

A conclusão final é gerada por um LLM de grande porte, utilizando
exclusivamente os trechos mais similares.

Características da conclusão:

-   Texto contínuo
-   Linguagem técnica
-   Múltiplos parágrafos
-   Avaliação crítica e comparativa
-   Alto grau de auditabilidade

------------------------------------------------------------------------

## 10. Estrutura dos Resultados

O resultado final contém:

-   **Conclusão técnica consolidada**
-   **Pares de trechos semanticamente semelhantes**, com score de
    similaridade

------------------------------------------------------------------------

## 11. Boas Práticas Adotadas

-   Quantização em 4 bits
-   Separação clara de responsabilidades
-   Modularidade
-   Reprodutibilidade
-   Preparação para evolução em arquiteturas RAG

------------------------------------------------------------------------

## 12. Possíveis Extensões

-   Persistência em banco vetorial
-   Integração com pipelines CI/CD
-   Classificação automática de maturidade DevOps
-   Interface web para visualização de similaridade
-   Análise longitudinal de documentos

------------------------------------------------------------------------

## 13. Conclusão Final

Este pipeline representa uma arquitetura madura e extensível de
análise semântica, adequada para documentação estratégica,
auditorias técnicas, governança de software e pesquisa
aplicada em engenharia de software.

------------------------------------------------------------------------
