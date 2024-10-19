import os
import KEYS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate

# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = KEYS.GOOGLE_API_KEY


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0
)


instructor_embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Represent the query for retrieval: "
)
vectordb_filepath="faiss_index_evis"

def create_vector_db():
    loader = CSVLoader(file_path=r"G:\sk_ai\merged_final.csv", source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data,embedding=instructor_embeddings)
    vectordb.save_local(vectordb_filepath)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_filepath, embeddings=instructor_embeddings,allow_dangerous_deserialization=True)

    # For local usage
    # Define a custom prompt template locally
    prompt_template = PromptTemplate(
        #input_variables=["context", "question"],
        input_variables=["company", "context", "question"],
        #template="You are a helpful assistant. Given the following context:\n\n{context}\n\nAnswer the question: {question}"
        template=(
            "You are a customer support representative for {company}. "
            "Your goal is to assist the customer with any issues they have, "
            "using the following context:\n\n{context}\n\n"
            "Please provide a helpful and polite response to the customer's question: {question}."
        )
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
            {
                "company": RunnablePassthrough(),
                "context": vectordb.as_retriever(score_threshold=0.7) | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt_template
            | llm
            | StrOutputParser()
    )
    return qa_chain

#if __name__ == "__main__":
    #create_vector_db()
    #qa_chain = get_qa_chain()
    #result = qa_chain.invoke("Do you have EMI option and intership ?")
    #print(result)