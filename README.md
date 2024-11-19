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

## Bibliotecas Necessárias

- **Transformers**: A *Transformers* é uma biblioteca da Hugging Face essencial para trabalhar com modelos de linguagem pré-treinados. Ela oferece ferramentas para tarefas de Processamento de Linguagem Natural (PLN), como geração de texto, tradução, classificação, entre outras.

- **einops**: A *einops* é uma biblioteca que oferece uma maneira simples e eficiente de manipular tensores de alta dimensão. Com ela, é possível realizar operações como reshape, permutação e redução de forma legível e otimizada, facilitando o trabalho com grandes modelos de linguagem.

- **accelerate**: A *accelerate* simplifica o processo de treinamento de modelos, especialmente em configurações com múltiplas GPUs. Ela permite paralelização e distribuição de tarefas, tornando o treinamento de modelos grandes mais rápido e acessível, sem a necessidade de configurações complexas.

- **bitsandbytes**: A *bitsandbytes* é uma biblioteca que permite a quantização de modelos de linguagem, reduzindo seu tamanho e exigências de memória sem sacrificar muita precisão. Isso é essencial para executar LLMs (Modelos de Linguagem de Grande Escala) de forma eficiente em máquinas com recursos limitados.


```python
!pip install -q transformers einops accelerate bitsandbytes gradio
```

## Hands On:
<b>Imports:</b> Para começar, devemos realizar as importações das bibliotecas necessárias para execução do código:
```bash
import getpass
import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, pipeline
```

**Verificação de GPU:** Se estiver executando o codigo localmente, será necessário realizar a configuração do Nvidia CUDA e cuDNN para que o Pytorch consiga acessar a placa de vídeo para realizar o processamento (verificar na documentação se sua placa de vídeo é compatível). Para este exemplo, estarei usando uma RTX 2060. O codigo abaixo deverá retornar o nome da placa de vídeo se disponível:
```python
device = "cuda:0" if torch.cuda.is_available() else "cpu" # Verificar se há gpu disponivel
print(torch.cuda.get_device_name())
```

**API KEY:** Para utilizar os modelos disponibilizados no HuggingFace é necessário realizar a criação da chave API, você poderá gerar sua chave após logar na sua conta do huggingFace acessando **Settings** > **Access Tokens** > **Create new token**:

![image](https://github.com/user-attachments/assets/e4674b4f-e475-45de-93d6-d71cefe522b7)

O codigo abaixo irá solicitar o Access Token gerado e irá configurar a variável de ambiente:
```python
os.environ["HF_TOKEN"] = getpass.getpass()
```

### Configuração e Download do Modelo:
Para este exemplo, iremos utilizar o modelo Phi-3.5-mini-instruct da microsoft, para isso, precisaremos especificar o modelo via `id` que é encontrado na pagina do modelo no huggingFace:

![image](https://github.com/user-attachments/assets/8ec17b13-11df-49da-a9ac-d59cfd80dd8c)

Com essa informação, executaremos o codigo abaixo para baixar o modelo
```python

# Identificador do modelo a ser baixado do repositório Hugging Face
id_model = 'microsoft/Phi-3.5-mini-instruct'

# Carrega o modelo selecionado a partir do repositório Hugging Face
model = AutoModelForCausalLM.from_pretrained(
    id_model,                    # Identificador único do modelo a ser baixado
    device_map="cuda",            # Define o uso de GPU (cuda) para aceleração de hardware
    torch_dtype="auto",           # Permite ao PyTorch selecionar automaticamente o tipo de dado (float32 ou float16) com base na disponibilidade de hardware
    trust_remote_code=True,       # Habilita a execução de código remoto, permitindo que o modelo baixe e execute scripts adicionais necessários para o seu funcionamento
    attn_implementation="eager"   # Define a implementação do mecanismo de atenção, sendo "eager" um modo mais simples e direto
)

# Baixa e carrega o tokenizador correspondente ao modelo, necessário para converter texto em tokens e vice-versa
tokenizer = AutoTokenizer.from_pretrained(id_model)

```

**Pipeline:** Após realizar o download do modelo, é necessário configurar a pipeline de execuções do modelo
```python
# Criação de um pipeline para execução de geração de texto, utilizando o modelo e tokenizador definidos anteriormente
pipe = pipeline(
    "text-generation",            # Define o tipo de tarefa do pipeline: "text-generation" (geração de texto)
    model=model,                  # Passa o modelo previamente carregado que será usado para a geração de texto
    tokenizer=tokenizer           # Passa o tokenizador previamente carregado, necessário para converter texto em tokens e vice-versa
)
```

***Parâmetros para geração de texto:*** Após configurar a estrutura do pipeline, é necessário passar as informações sobre como o texto será gerado, para isso utilizaremos a definição abaixo:
```python
generation_args = {
    "max_new_tokens": 500,  # Máximo de tokens gerados.
    "return_full_text": False,  # Retornará apenas a resposta gerada.
    "temperature": 0.1,  # Controle da criatividade (menor = mais direto).
    "do_sample": True,  # Respostas mais aleatórias.
}

```
**Utilização:** Agora que quase todas configurações foram feitas, vamos criar uma função para realizar as requisições de respostas ao modelo, para isso utilizaremos também o gradio, para forncecer uma interface amigável de iteração com chatbot, para isso, utilizamos o codigo abaixo:
```python
def realizar_pergunta(prompt):
    messages = [
        {"role": "system", "content" : "Você é um assistente virtual prestativo. Responda somente em Português."},
        {"role": "user", "content": prompt}
    ]
    
    output = pipe(messages, **generation_args)
    return output[0]['generated_text']

# Função para a interface Gradio
def chat_interface(prompt):
    resposta = realizar_pergunta(prompt)
    return resposta[2]['content']

# Criando a interface Gradio
iface = gr.Interface(fn=chat_interface, inputs="text", outputs="text", live=True, title="Assistente Virtual")

# Iniciando a interface
iface.launch()
```

O codigo acima irá criar um servidor com a interfacio Gradio, que possibilita a iteração com o modelo, podemos ver conforme imagem abaixo o teste com a seguinte pergunta: `O que são LLMs?`:

![image](https://github.com/user-attachments/assets/ae72ed7f-5e08-4a58-936b-1d72deda91b2)

## Conclusão
Neste repositório, mostramos como utilizar LLMs com a Hugging Face de maneira simples e eficiente, usando Python. Agora você tem os conhecimentos básicos para aplicar modelos de linguagem em seus próprios projetos e explorar as possibilidades dessa tecnologia.
