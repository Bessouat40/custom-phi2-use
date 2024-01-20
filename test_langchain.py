#tester online + créer bdd vectorisée (context)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

#pip install --upgrade --quiet  llama-cpp-python --no-cache-dirclear

llm = LlamaCpp(
    model_path="./llama-2-13b.gguf.q6_K.bin",
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

llm("The first man on the moon was ... Let's think step by step")