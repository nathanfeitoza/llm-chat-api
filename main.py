from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from transformers import pipeline
from huggingface_hub import hf_hub_download, login
from llama_cpp import Llama
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import json
from concurrent.futures import ThreadPoolExecutor
import asyncio

model_path = "/mnt/b/LLM/models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

executor = ThreadPoolExecutor(max_workers=10)

model_clusters = []
max_model_clusters = 3

class Model:
    _model_id = 0
    _model = None
    _in_use = False
    
    def __init__(self, model_id = None, model = None, in_use = False):
        self._model = model
        self._model_id = model_id
        self._in_use = in_use
    
    @staticmethod
    def create(model_id):
        model_use = Llama(model_path=model_path, n_gpu_layers=50, quantization="q4_1", n_threads=5)
        return Model(model_id, model_use, False)
    
    def get_model(self):
        return self._model
    
    def using(self):
        return self._in_use
        
    def use(self):
        self._in_use = True
        return self
        
    def free(self):
        self._in_use = False
        return self

class Models:
    _models = []
    
    def __init__(self):
        pass

    def load(self):
        for i in range(max_model_clusters):
            print("Create model llm: ", i)
            self._models.append(Model.create(i))
        
        return self
    
    def get_model(self):
        response = [model for model in self._models if model.using() == False]
        
        return response
            
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",  # Permitir essas origens
    allow_credentials=True,  # Permitir envio de cookies e credenciais
    allow_methods=["*"],  # Permitir todos os métodos HTTP (GET, POST, etc)
    allow_headers=["*"],  # Permitir todos os cabeçalhos
)

# Função para rodar o modelo LLaMA de forma síncrona
def run_llama_model(model_loaded, messages, max_tokens=100, temperature=0.7, stream=True):
    return model_loaded.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream  # Habilitar o streaming no modelo
    )

# Definindo o modelo para cada mensagem no array
class MessageModel(BaseModel):
    role: str
    content: str

# Definindo o modelo de entrada que contém o array de mensagens e outros parâmetros
class InputModel(BaseModel):
    messages: List[MessageModel]  # Lista de objetos MessageModel
    temperature: float = 0.7
    max_tokens: int = -1
    stream: bool = True

# Função que gera chunks de dados para o EventStream
async def stream_response(request: InputModel, model_loaded):
    loop = asyncio.get_event_loop()
    
    response = await loop.run_in_executor(
        executor, run_llama_model, model_loaded, request.messages, request.max_tokens, request.temperature, request.stream
    )

    # Processa e envia os pedaços de resposta ao cliente em markdown
    for chunk in response:
        chat_id = chunk['id']
        choices = chunk['choices']
        response_json = json.dumps({ 'id': chat_id, 'choices': choices })
        print("response_json", response_json)

        # Encerra o evento
        if chat_id == "DONE":
            break
        
        yield f"data: {response_json}\n\n"
    
    yield "data: [DONE]\n\n"  # Indica o fim do stream

models_loades = Models().load()

# Rota para processar a entrada e retornar via EventStream
@app.post("/conversation")
async def process_input(input_model: InputModel):
    model_load = models_loades.get_model()
    
    if (len(model_load) == 0):
        raise HTTPException(
            status_code=412,
            detail="All models are currently busy. Please try again later."
        )
    
    model_loaded = model_load[-1]
    
    model_loaded.use()
    response = StreamingResponse(stream_response(input_model, model_loaded.get_model()), media_type="text/event-stream")
    model_loaded.free()
    # Retorna a resposta como EventStream
    return response
