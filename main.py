import streamlit as st
from langchain_helper import create_vector_db,get_qa_chain

st.title("EVIS Virtual Assistant")
# btn = st.button("UPDATE Knowledgebase")
# if btn:
#     pass

question = st.text_input("",placeholder="Ask anything about our products .. ")

if question:
    qa_chain = get_qa_chain()
    response = qa_chain.invoke(question)

    st.header("Answer: ")
    st.write(response)

# if __name__ == "__main__":
#     create_vector_db()

