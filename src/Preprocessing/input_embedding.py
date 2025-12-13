# from langchain_community.embeddings import HuggingFaceEmbeddings

# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
#     # encode_kwargs={"normalize_embeddings": True}
# )


from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    # encode_kwargs={"normalize_embeddings": True}
)