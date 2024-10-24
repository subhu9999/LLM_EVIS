import streamlit as st
from langchain_helper import get_qa_chain
from nltk_helper import is_question

st.title("EVIS Virtual Assistant")
# btn = st.button("UPDATE Knowledgebase")
# if btn:
#     pass

# Initialize question history in session state if it doesn't exist
if 'question_history' not in st.session_state:
    st.session_state.question_history = []
# Initialize question history as a list
# question_history = []

question = st.text_input("",placeholder="Ask anything about our products .. ")

def update_history(response,question_history):
    if(is_question(response)):
        return [question_history[-1]]
    else:
        return []

if question:
    # st.write(st.session_state.question_history)
    qa_chain = get_qa_chain()
    # response = qa_chain.invoke(question)
    st.header("Answer: ")
    # st.write(response)
    st.session_state.question_history.append(question)
    question_string = ', '.join(st.session_state.question_history)
    # Invoke the chain with updated question history
    response = qa_chain.invoke(question_string)
    st.write(response)
    # question_history = update_history(response, st.session_state.question_history)
    st.session_state.question_history = update_history(response, st.session_state.question_history)
    # st.write(st.session_state.question_history)
# if __name__ == "__main__":
#     create_vector_db()

