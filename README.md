# Consumindo Modelos de Linguagem com Hugging Face e Python

# Introdução
Com o avanço da Inteligencia Artificial (IA) e do processamento de linguagem natural (PLN), os Modelos de Linguagm de Grande Escala (LLMs) têm ganhado cada vez mais destaque, tornando-se cada vez mais essenciais em diversas aplicações de IA no mundo real.
A Hugging Face é uma plataforma que oferece diversas ferramentas poderosas para se trabalhar com esses modelos de linguagem, fornecendo acesso a uma vasta gama de modelos pré-treinados que facilita a integração desses modelos e aplicações de software. O Python, por ser uma linguagem flexível e com grande suporte para IA, torna-se a escolha ideal para consumir e interagir com esses modelos.


## Objetivo
O objetivo deste repositório é mostrar como consumir modelos de linguagem de grande escala (LLMs) de forma simples e rápida, utilizando a biblioteca *Transformers* da Hugging Face e Python. A ideia é proporcionar um guia direto e acessível para que qualquer desenvolvedor ou pesquisador consiga integrar esses poderosos modelos em suas próprias aplicações, sem a necessidade de infraestrutura complexa.

###  ⚠️ Importância do uso de GPU
Embora seja possível rodar alguns desses modelos em CPUs, o uso de **GPU** (Unidade de Processamento Gráfico) é altamente recomendado e essencial para um desempenho eficiente. Com o uso de uma GPU, o tempo de inferência e o uso de memória são significativamente reduzidos, tornando o processo de execução muito mais rápido e escalável. Para aqueles que não possuem acesso a uma GPU, o **Google Colab** oferece uma opção gratuita de uso de GPU, permitindo que você execute seus modelos de linguagem de forma mais rápida e eficiente. Basta criar uma conta no Colab e selecionar a opção de GPU em **Ambiente de Execução** > **Alterar tipo de hardware** > **GPU** (conforme disponibilidade).


## Instalando as Dependências
Antes de começarmos a utilizar os modelos de linguagem, precisamos instalar algumas bibliotecas que são essenciais para execução do modelo:

## Bibliotecas Importadas

- **Transformers**: A *Transformers* é uma biblioteca da Hugging Face essencial para trabalhar com modelos de linguagem pré-treinados. Ela oferece ferramentas para tarefas de Processamento de Linguagem Natural (PLN), como geração de texto, tradução, classificação, entre outras.

- **einops**: A *einops* é uma biblioteca que oferece uma maneira simples e eficiente de manipular tensores de alta dimensão. Com ela, é possível realizar operações como reshape, permutação e redução de forma legível e otimizada, facilitando o trabalho com grandes modelos de linguagem.

- **accelerate**: A *accelerate* simplifica o processo de treinamento de modelos, especialmente em configurações com múltiplas GPUs. Ela permite paralelização e distribuição de tarefas, tornando o treinamento de modelos grandes mais rápido e acessível, sem a necessidade de configurações complexas.

- **bitsandbytes**: A *bitsandbytes* é uma biblioteca que permite a quantização de modelos de linguagem, reduzindo seu tamanho e exigências de memória sem sacrificar muita precisão. Isso é essencial para executar LLMs (Modelos de Linguagem de Grande Escala) de forma eficiente em máquinas com recursos limitados.



- ** 
```bash
!pip install -q transformers einops accelerate bitsandbytes
