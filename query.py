

# re-order docs after retrieval to avoid performance degradation:
# https://python.langchain.com/docs/modules/data_connection/document_transformers/post_retrieval/long_context_reorder

# prepend the following instruction to the user query
instruction = "Represent this sentence for searching relevant passages: "