
from langchain_openai import ChatOpenAI
import os
import re


# Map each property code to the names staff may use for it (code, city), built from index metadata
def build_property_aliases(vectorstore):
    aliases = {}
    for doc in vectorstore.docstore._dict.values():
        code = doc.metadata.get("property")
        if code:
            names = aliases.setdefault(code, {code.lower()})
            if doc.metadata.get("city"):
                names.add(doc.metadata["city"].lower())
    return aliases


# Return the property codes mentioned in the query by code or city name
def detect_properties(query, aliases):
    q = query.lower()
    return {code for code, names in aliases.items()
            if any(re.search(rf"\b{re.escape(n)}\b", q) for n in names)}


# Retrieve the top-k relevant chunks, guaranteeing rows for any property named in the query
def retrieve(query: str, k: int = 6, vectorstore=None, property_boost: bool = True) -> list[str]:
    results = []
    if property_boost:
        codes = detect_properties(query, build_property_aliases(vectorstore))
        if codes:
            # Filtered search over ALL vectors (fetch_k=ntotal); FAISS filters only after fetching
            results = vectorstore.similarity_search(
                query, k=2 * len(codes),
                filter=lambda md: md.get("property") in codes,
                fetch_k=vectorstore.index.ntotal,
            )

    # Add the normal top-k semantic results, skipping duplicates
    seen = {d.page_content for d in results}
    for d in vectorstore.similarity_search(query, k=k):
        if d.page_content not in seen:
            results.append(d)
            seen.add(d.page_content)

    # Return each chunk with its content and metadata
    return [f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}" for r in results]


# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks, model) -> str:

    rules = """
    1. Answer using ONLY the Retrieved Policy Information above. Do not use outside knowledge or assumptions.
    2. If the retrieved information answers only part of the question, answer the parts it supports and state exactly which part is not covered. If nothing relevant was retrieved, say which item (e.g., the property, room type, or policy topic) is not covered in the available policy documents. Do not add next steps that are not stated in the retrieved information.
    3. Answer the guest's question directly in the first sentence, then give supporting details.
    4. When a rule is property-specific, use only the fee or rule for the property named in the query. Never apply one property's fee to another.
    5. Include relevant conditions, limits, caps, and exceptions (e.g., stay caps, service animal exemptions, availability limits).
    6. When the question involves a calculation (e.g., fees for several nights), show the calculation and apply any caps.
    7. Answer every part of a multi-part question.
    8. Cite the source document name(s) the answer is based on.
    9. Keep the answer concise and professional, suitable for hotel staff to relay to a guest.
    """

    prompt = f"""
    You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    {rules}

    Answer:
    """

    return model.invoke(prompt)


# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 6,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
