"""
interactive function is used to interactively run the pipeline for a given question. It takes the following arguments:
question: str: The question for which the pipeline needs to be run
vectorstore: Chroma: The vectorstore object
node_context_df: pd.DataFrame: The dataframe containing the context for each node
embedding_function_for_context_retrieval: SentenceTransformer: The sentence transformer model for embedding the context
llm_type: str: The type of LLM model to be used
edge_evidence: bool: Flag to show evidence of association from the graph
system_prompt: str: The system prompt to be used
api: bool: Flag to use the API for context retrieval
llama_method: str: The method to choose for the Llama model
"""
from kg_rag_keppylab.config_loader import system_prompts, config_data
from kg_rag_keppylab.GPT import disease_entity_extractor_GPT, get_GPT_response
import click

@click.command()
@click.option('--question', help='The question for which the pipeline needs to be run')



def interactive(question, vectorstore, node_context_df, embedding_function_for_context_retrieval, llm_type, edge_evidence, system_prompt, api=True, llama_method="method-1"):
    print(" ")
    input("Press enter for Step 1 - Disease entity extraction using GPT-3.5-Turbo")
    print("Processing ...")
    entities = disease_entity_extractor_GPT(question)
    max_number_of_high_similarity_context_per_node = int(config_data["CONTEXT_VOLUME"]/len(entities))
    print("Extracted entity from the prompt = '{}'".format(", ".join(entities)))
    print(" ")

    input("Press enter for Step 2 - Match extracted Disease entity to SPOKE nodes")
    print("Finding vector similarity ...")
    node_hits = []
    for entity in entities:
        node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
        node_hits.append(node_search_result[0][0].page_content)
    print("Matched entities from SPOKE = '{}'".format(", ".join(node_hits)))
    print(" ")

    input("Press enter for Step 3 - Context extraction from SPOKE")
    node_context = []
    for node_name in node_hits:
        if not api:
            node_context.append(node_context_df[node_context_df.node_name == node_name].node_context.values[0])
        else:
            context, context_table = get_context_using_spoke_api(node_name)
            node_context.append(context)
    print("Extracted Context is : ")
    print(". ".join(node_context))
    print(" ")

    input("Press enter for Step 4 - Context pruning")
    question_embedding = embedding_function_for_context_retrieval.embed_query(question)
    node_context_extracted = ""
    for node_name in node_hits:
        if not api:
            node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
        else:
            node_context, context_table = get_context_using_spoke_api(node_name)
        node_context_list = node_context.split(". ")
        node_context_embeddings = embedding_function_for_context_retrieval.embed_documents(node_context_list)
        similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
        similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
        percentile_threshold = np.percentile([s[0] for s in similarities], config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
        high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"]]
        if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
            high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
        high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
        if edge_evidence:
            high_similarity_context = list(map(lambda x:x+'.', high_similarity_context))
            context_table = context_table[context_table.context.isin(high_similarity_context)]
            context_table.loc[:, "context"] =  context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype('str') + "\n\n"
            node_context_extracted = context_table.context.str.cat(sep=' ')
        else:
            node_context_extracted += ". ".join(high_similarity_context)
            node_context_extracted += ". "
    print("Pruned Context is : ")
    print(node_context_extracted)
    print(" ")

    input("Press enter for Step 5 - LLM prompting")
    print("Prompting ", llm_type)
    if llm_type == "llama":
        from langchain import PromptTemplate, LLMChain
        template = get_prompt("Context:\n\n{context} \n\nQuestion: {question}", system_prompt)
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        llm = llama_model(config_data["LLAMA_MODEL_NAME"], config_data["LLAMA_MODEL_BRANCH"], config_data["LLM_CACHE_DIR"], stream=True, method=llama_method)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        output = llm_chain.run(context=node_context_extracted, question=question)
    elif "gpt" in llm_type:
        enriched_prompt = "Context: "+ node_context_extracted + "\n" + "Question: " + question
        output = get_GPT_response(enriched_prompt, system_prompt, llm_type, llm_type, temperature=config_data["LLM_TEMPERATURE"])
        stream_out(output)
