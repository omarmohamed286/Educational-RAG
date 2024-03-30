from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.chains.conversation.memory import ConversationBufferMemory



model_path = 'openchat_3.5.Q3_K_L.gguf'
n_gpu_layers = 1 
n_batch = 512 
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


model = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=callback_manager,
    verbose=True,
)


#answered totally wrong, very strong hallucination
print(model("Q: how does hnsw algorithm work? A: "))

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector-db")

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

retriever = db.as_retriever(search_kwargs={"k":5})


chain_type_kwargs = {"prompt": prompt}
qa = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
query = "how does hnsw algorithm work?"
response = qa(query)
print(response)



