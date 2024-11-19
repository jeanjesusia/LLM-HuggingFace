# Importando libs
import getpass
import os
import torch
import warnings
warnings.filterwarnings("ignore")

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, pipeline


# Configuração de GPU
print(torch.cuda.get_device_name())
device = "cuda:0" if torch.cuda.is_available() else "cpu" # Verificar se há gpu disponivel
torch.random.manual_seed(42)


# Configurando Access Token
os.environ["HF_TOKEN"] = getpass.getpass()


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


# Criação de um pipeline para execução de geração de texto, utilizando o modelo e tokenizador definidos anteriormente
pipe = pipeline(
    "text-generation",            # Define o tipo de tarefa do pipeline: "text-generation" (geração de texto)
    model=model,                  # Passa o modelo previamente carregado que será usado para a geração de texto
    tokenizer=tokenizer           # Passa o tokenizador previamente carregado, necessário para converter texto em tokens e vice-versa
)


# Configurando Parâmetros para geração de Texto
generation_args = {
    "max_new_tokens": 500,  # Máximo de tokens gerados.
    "return_full_text": False,  # Retorna apenas a resposta.
    "temperature": 0.1,  # Controle da criatividade (menor = mais direto).
    "do_sample": True,  # Respostas mais aleatórias.
}

# Função para realizar requisições de respostas ao modelo
def realizar_pergunta(prompt):
    messages = [
        {"role": "system", "content": "Você é um assistente virtual prestativo. Responda somente em Português."},
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
