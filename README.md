# 📘 Documentação Técnica Completa

## Pipeline de Análise Semântica, Releases e Comparação Textual com LLMs

------------------------------------------------------------------------

## 🧭 Sumário

1.  Visão Geral\
2.  Objetivos do Sistema\
3.  Modelos Utilizados e Motivação\
4.  Arquitetura Geral\
5.  Dependências e Requisitos\
6.  Pipeline de Análise de Releases\
7.  Pipeline Alternativo com Modelo Leve\
8.  Pipeline de Comparação Semântica\
9.  Geração de Conclusão Técnica\
10.  Estrutura dos Resultados\
11. Boas Práticas Adotadas\
12. Possíveis Extensões\
13. Conclusão Final

------------------------------------------------------------------------

## 1. Visão Geral

Este projeto implementa um **pipeline avançado de análise semântica
aplicada à engenharia de software**, combinando **Modelos de Linguagem
de Grande Porte (LLMs)** com **modelos de embeddings vetoriais** para
análise técnica profunda de artefatos textuais.

O sistema foi projetado para execução local com **aceleração por GPU**,
utilizando **quantização em 4 bits**, permitindo alto desempenho com uso
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

BAAI/bge-base-en-v1.5 (BGE-Base):

Motivação: Escolhido por ser um modelo de embedding de alta eficiência. Ele permitiu a criação de vetores de similaridade semântica para os logs de release de forma rápida, sendo fundamental para o funcionamento do pipeline de comparação de documentos sem sobrecarregar a memória RAM.

Qwen 2.5 (1.5B-Instruct):

Motivação: Atuou como modelo de apoio inicial para contextualização. Devido à nossa infraestrutura inferior ao que seria necessário para modelos gigantes, o Qwen foi a escolha ideal por ser leve (1.5B), permitindo a triagem rápida dos dados do Screenpipe com baixo consumo de recursos.

Mistral-7B-Instruct-v0.3:

Motivação: Responsável pela síntese e consolidação técnica. Para rodar em nossa GPU (RTX 4060) ou Colab T4, utilizamos quantização de 4 bits, o que permitiu extrair o conhecimento técnico do modelo sobre o fluxo de trabalho (Workflows) sem exceder a VRAM disponível.

Meta-LLaMA 3.1-8B-Instruct:

Motivação: Utilizado para a avaliação crítica final e validação das estratégias de Engenharia de Software. Sua escolha justifica-se pelo alto grau de raciocínio lógico, sendo capaz de confirmar a aderência ao GitHub Flow e Rapid Releases, funcionando como o "validador" final de todo o processo.
------------------------------------------------------------------------

## 4. Arquitetura Geral

A arquitetura é organizada em **três camadas lógicas**, com
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

-   transformers\
-   accelerate\
-   bitsandbytes\
-   torch\
-   sentence-transformers\
-   numpy

### 5.2 Requisitos de Hardware

-   GPU NVIDIA com suporte CUDA\
-   Recomendado **≥ 8 GB de VRAM** para execução do Mistral 7B em 4-bit

------------------------------------------------------------------------

## 6. Pipeline de Análise de Releases

Este pipeline processa logs de releases para identificar:

-   Estratégia de versionamento\
-   Cadência de entregas\
-   Modelo de workflow (GitFlow, Trunk-based, híbrido)

Utiliza prompts estruturados no formato nativo do modelo, com geração
determinística e controle rigoroso de contexto.

------------------------------------------------------------------------

## 7. Pipeline Alternativo com Modelo Leve

O modelo **Qwen 1.5B** é utilizado como alternativa ultraleve, indicado
para:

-   Ambientes com restrição de VRAM\
-   Execuções rápidas\
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

-   Texto contínuo\
-   Linguagem técnica\
-   Múltiplos parágrafos\
-   Avaliação crítica e comparativa\
-   Alto grau de auditabilidade

------------------------------------------------------------------------

## 10. Estrutura dos Resultados

O resultado final contém:

-   **Conclusão técnica consolidada**
-   **Pares de trechos semanticamente semelhantes**, com score de
    similaridade

------------------------------------------------------------------------

## 11. Boas Práticas Adotadas

-   Quantização em 4 bits\
-   Separação clara de responsabilidades\
-   Modularidade\
-   Reprodutibilidade\
-   Preparação para evolução em arquiteturas RAG

------------------------------------------------------------------------

## 12. Possíveis Extensões

-   Persistência em banco vetorial\
-   Integração com pipelines CI/CD\
-   Classificação automática de maturidade DevOps\
-   Interface web para visualização de similaridade\
-   Análise longitudinal de documentos

------------------------------------------------------------------------

## 13. Conclusão Final

Este pipeline representa uma **arquitetura madura e extensível de
análise semântica**, adequada para **documentação estratégica**,
**auditorias técnicas**, **governança de software** e **pesquisa
aplicada em engenharia de software**.

------------------------------------------------------------------------
