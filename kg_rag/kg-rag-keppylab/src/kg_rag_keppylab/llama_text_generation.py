import click
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chroma import Chroma
from langchain.utils import cosine_similarity

SYSTEM_PROMPT = system_prompts["KG_RAG_BASED_TEXT_GENERATION"]
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]

def load_sentence_transformer(sentence_embedding_model):
    return SentenceTransformerEmbeddings(model_name=sentence_embedding_model)

def load_chroma(vector_db_path, sentence_embedding_model):
    embedding_function = load_sentence_transformer(sentence_embedding_model)
    return Chroma(persist_directory=vector_db_path, embedding_function=embedding_function)

@click.command()
@click.option('--question', help='The question for which the pipeline needs to be run', required=True, type=str)
@click.option('--llama_method', help='The method to choose for the Llama model', required=True, type=str, default='method-1')
@click.option('--edge_evidence', help='Flag for showing evidence of association from the graph', required=False, type=bool, default=False)
def interactive(question: str, llama_method: str, edge_evidence: bool, system_prompt: str):
    """
    This function is used to run the pipeline in an interactive mode.

    """
    vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
    embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
    node_context_df = pd.read_csv(NODE_CONTEXT_PATH)

    print(" ")
    input("Press enter for Step 1 - Disease entity extraction using GPT-3.5-Turbo")
    print("Processing ...")
    entities = disease_entity_extractor_GPT(question)
    max_number_of_high_similarity_context_per_node = int(config_data["CONTEXT_VOLUME"]/len(entities))
    print("Extracted entity from the prompt = '{}'".format(", ".join(entities)))
    print(" ")

if __name__ == "__main__":
    interactive()
