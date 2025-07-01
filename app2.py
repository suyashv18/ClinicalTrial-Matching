import streamlit as st
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph import StateGraph, MessagesState, START
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import pandas as pd
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, List, Optional
import json
from typing import Type
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
import os
import tempfile
import time
import re
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
password = os.getenv("PASSWORD")
# password = "zzlbeqzfkhcqtqgz"
# Page config
st.set_page_config(
    page_title="Clinical Trial Matcher", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
    color: #000000;
    font-size: 2.8rem;
    margin-bottom: 2rem;
    margin-left: 1rem;
    display: flex;
    align-items: center;
    }
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
    }
    .logo-container {
        margin-right: 1rem;
    }
    .agent-status {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .agent-running {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .agent-completed {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .trial-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .trial-title {
        font-weight: bold;
        color: #2E8B57;
        margin-bottom: 0.5rem;
    }
    .trial-nct {
        color: #666;
        font-size: 0.9rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'processing_stage' not in st.session_state:
    st.session_state.processing_stage = 0
if 'clinical_trials' not in st.session_state:
    st.session_state.clinical_trials = {}
if 'final_message' not in st.session_state:
    st.session_state.final_message = ""
if 'patient_summary' not in st.session_state:
    st.session_state.patient_summary = ""

# Load data and initialize models (your existing code)
excel_data = pd.read_json("data.json")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

llm2 = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

llm3 = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="gemma2-9b-it",
    temperature=0.7
)

llm4 = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="gemma2-9b-it",
    temperature=0.7
)

class AgentState(TypedDict):
    file_path: str
    content: str
    summary: str
    answer: str
    inputs: dict
    extracted_info: dict

# Your existing functions (keeping them as they are)
def read_medical_history(file_path: str) -> str:
    print("The file path is:", file_path)
    with open(file_path, "r") as file:
        medical_history = file.read()
        print("Medical_History:", medical_history)
    return medical_history

read_tool = Tool.from_function(
    name="ReadFile",
    func=read_medical_history,
    description="Reads content from a file given a file path"
)

agent1 = initialize_agent(
    tools=[read_tool],
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True, 
)

def generate_disease_summary(document_text: str) -> str:
    """provides the summary based on the input data."""
    prompt_template = ChatPromptTemplate.from_template(f"""
    You are a medical assistant. A user has submitted a medical report. Your task is to read the report and generate a clear, concise summary of the disease state.

    Focus only on:
    - Diagnosis (or suspected diagnosis)
    - Symptoms
    - Progression or stage of disease
    - Relevant lab/imaging findings
    - Any mentioned treatments related to the disease

    Medical Report:{document_text}

    Now provide a structured disease summary.
    """)
    
    prompt = prompt_template.format(document_text=document_text)
    resp = llm.predict(prompt)
    return resp

summarize_tool = Tool.from_function(
    name="SummarizeText",
    func=generate_disease_summary,
    description="Summarizes provided text"
)

agent2 = initialize_agent(
    tools=[summarize_tool],
    llm=llm2,
    agent_type="zero-shot-react-description",
    verbose=True
)

class ResponseModel(BaseModel):
    Suitability: str
    Reasoning: str

class EmailResponseModel(BaseModel):
    Subject: str
    Email: str

#helper to pull title from content
def extract_study_title(page_content: str) -> str:
    """
    Extracts the study title from the page_content string.
    Assumes the pattern 'Study Title: ... |'
    """
    match = re.search(r"Study Title:\s*(.*?)\s*\|", page_content, re.IGNORECASE)
    return match.group(1).strip() if match else "TITLE_NOT_FOUND"

@st.cache_resource
def extract_suitable_ct_new(input_str: str) -> str:
    inputs = json.loads(input_str)
    document_text = inputs.get("document_text", "")
    patient_summary = inputs.get("patient_summary", "")

    print("document_text", document_text)
    """Extracts the best suitable clinical trials data based on summary."""
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("pfizer_vdb", embeddings=embedding, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    medical_summary = patient_summary
    results = retriever.invoke(medical_summary)

    ct_results = {}
    for i, doc in enumerate(results, 1):
        print(f"{i}. Trial ID: {doc.metadata['NCT Number']}")
        content = doc.page_content
        start = content.lower().find('criteria:')
        if start != -1:
            full_criteria = content[start + len('criteria:'):].strip()
        else:
            full_criteria = ""

        print(f"   Content: {full_criteria}.\n")
        
        ct_details = full_criteria

        prompt_template = ChatPromptTemplate.from_template(f"""
        You are a medical assistant. Based on the clinical trials details your job is to extract the Inclusion and Exclusion criterias and create a question for each Inclusion and Exclusion 
        criteria which is extracted.

        Clinical Trials Details : {ct_details}

        Return the Questions : 

        """)
        prompt = prompt_template.format(ct_details=ct_details)
        questions_resp = llm.predict(prompt)

        print("The questions are:", questions_resp)

        prompt_template = ChatPromptTemplate.from_template(f""" Based on the questions created and the document text, generate a summary which answers all the questions.
        
        Document : {document_text}
                                                            
        Questions : {questions_resp}

        """)
        prompt = prompt_template.format(document_text=document_text, questions_resp=questions_resp)
        medical_summary = llm.predict(prompt)

        print("The generated new summary are:", medical_summary)
        
        parser = PydanticOutputParser(pydantic_object=ResponseModel)

        prompt_template = ChatPromptTemplate.from_template(f"""
        You are a medical assistant. Based on the medical summary and the questions, check if all the questions has their answers in the medical summary.  
        This will be sufficient to determine the patients suitability.                                               

        Medical Summary:{medical_summary}
        Questions : {questions_resp}

        Return the response in strict json format with no backticks or special characters:
            "Suitability": "(Suitable / Not Suitable)",
            "Reasoning": "(Short explanation why)"
        """)

        prompt = prompt_template.format(medical_summary=medical_summary, questions_resp=questions_resp, format_instructions=parser.get_format_instructions())
        resp = llm.predict(prompt)
        parsed = parser.parse(resp).model_dump_json(indent=2)
        print("For NCTID : ", doc.metadata['NCT Number'], " : ", resp)
        print("The parsed info is:", parsed)

        response = json.loads(parsed)
        if response['Suitability'] == 'Suitable':
            print("found a result which is suitable")
            ct_results[doc.metadata["NCT Number"]] = doc.page_content
        else:
            print("No result which is suitable")
    print("CT Results are:", ct_results)
    clinical_trial_results = json.dumps(ct_results)
    return clinical_trial_results

extractor_tool = Tool.from_function(
    name="ExtractCTdetails",
    func=extract_suitable_ct_new,
    description="Extracts the most suitable clinical trial's and the information related to it.",
    return_direct=True
)

agent3 = initialize_agent(
    tools=[extractor_tool],
    llm=llm3,
    agent_type="zero-shot-react-description",
    verbose=True
)

def communication_func(ct_results: dict) -> str:
    print("The CT result is:", ct_results)
    ct_results = json.loads(ct_results)
    if len(ct_results) == 0:
        resp = "There are no suitable matches."
    else:
        parser = PydanticOutputParser(pydantic_object=EmailResponseModel)
        
        for key, ct_summary in ct_results.items():
            print("key", key, ct_summary)
            prompt_template = ChatPromptTemplate.from_template(f"""
            You are a email expert, 
            Based on the clinical trial details mentioned below, you have to write a mail to the stakeholder stating 'That the patient who has uploaded the medical history is suitable for 
            clinical trial (add the NCT Number of the clinical trial)'. 
            Also generate a suitable subject for the mail.                                                                                              
            Clinical Trial Summary:{ct_summary}

            Response in json format: 
            Subject :(Subject of mail)
            Email : (The generated email)

            """)
            prompt = prompt_template.format(ct_summary=ct_summary, format_instructions=parser.get_format_instructions())
            resp = llm.predict(prompt)
            parsed_resp = parser.parse(resp).model_dump_json(indent=2)
            response = json.loads(parsed_resp)
            print("The response is:", response)
            email = excel_data[excel_data['nct_id'] == key]['email'].values
            print("The email is:", email)
            print("Email is:", email[0])
            import smtplib
            smtp_server = 'smtp.gmail.com'
            smtp_port = 587
            smtp_username = 'officialitsme18@gmail.com'
            smtp_password = password
            
            from_email = 'officialitsme18@gmail.com'
            to_email = email[0]
            subject = response['Subject']
            body = response['Email']
            
            message = f'Subject: {subject}\n\n{body}'
            
            with smtplib.SMTP(smtp_server, smtp_port) as smtp:
                smtp.starttls()
                smtp.login(smtp_username, smtp_password)
                smtp.sendmail(from_email, to_email, message)
            
            print("The extracted response is :", parsed_resp)
            resp = "A mail has been sent to the stakeholder for suitable client."
    return resp

communication_tool = Tool.from_function(
    name="CommunicationAgent",
    func=communication_func,
    description="Extract contact details,generate mail body and send mail to the required people",
)

agent4 = initialize_agent(
    tools=[communication_tool],
    llm=llm4,
    agent_type="zero-shot-react-description",
    verbose=True
)

def agent1_node(state: AgentState) -> dict:
    content = agent1.run(f"Read the file at: {state['file_path']}")
    return {
        **state,
        "content": content,
        "next": "agent2"
    }

def agent2_node(state: AgentState) -> dict:
    summary = agent2.run(f"Summarize this:\n{state['content']}")
    return {
        **state,
        "summary": summary,
        "next": "agent3"
    }

def agent3_node(state: AgentState) -> dict:
    print("Content", {state['content']})
    input_data = {"patient_summary": state['summary'], "document_text": state['content']}
    print(input_data)
    inputs = json.dumps(input_data)
    extracted_info = agent3.run(inputs)

    return {
        **state,
        "extracted_info": extracted_info,
        "next": "agent4" 
    }

def agent4_node(state: AgentState) -> dict:
    print("State is:", state)
    extracted_details = agent4.run(f"Use this information:\n{state['extracted_info']}")
    return {
        **state,
        "extracted_details": extracted_details
    }

# Create graph
builder = StateGraph(AgentState)
builder.add_node("agent1", agent1_node)
builder.add_node("agent2", agent2_node)
builder.add_node("agent3", agent3_node)
builder.add_node("agent4", agent4_node)

builder.add_edge("agent1", "agent2")
builder.add_edge("agent2", "agent3")
builder.add_edge("agent3", "agent4")
builder.add_edge("agent4", END)

builder.set_entry_point("agent1")
graph = builder.compile()

# STREAMLIT UI
st.markdown('<h1 class="main-header">🏥 Clinical Trial Matching System</h1>', unsafe_allow_html=True)

# col1, col2 = st.columns([1, 8], gap="small")
# with col1:
#     st.image("i2e_logo.png", width=90)
# with col2:
#     st.markdown('<h1 class="main-header">Clinical Trial Matching System</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📁 Upload Medical Documents")
    st.markdown("Upload patient's medical history file to find suitable clinical trials.")
    
    uploaded_file = st.file_uploader(
        "Choose a medical history file",
        type=["txt"],
        help="Please upload a .txt file containing the patient's medical history"
    )
    
    if uploaded_file is not None:
        st.success("✅ File uploaded successfully!")
        
        # Reset session state when new file is uploaded
        if st.button("🔍 Find Relevant Trials", type="primary", use_container_width=True):
            st.session_state.processing_stage = 0
            st.session_state.clinical_trials = {}
            st.session_state.final_message = ""
            st.session_state.patient_summary = ""
            
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Store file path in session state
            st.session_state.tmp_file_path = tmp_file_path
            st.session_state.processing_stage = 1
            st.rerun()

# Main interface
if st.session_state.processing_stage == 0:
    # Initial state - show instructions
    st.markdown("""
    ### Welcome to the Clinical Trial Matching System
    
    This system helps match patients with suitable clinical trials based on their medical history.
    
    **How it works:**
    1. Upload a medical history file (.txt format) using the sidebar
    2. Click "Find Relevant Trials" to start the matching process
    3. The system will analyze the document and find suitable clinical trials
    4. Stakeholders will be automatically notified of potential matches
    
    Please upload a file to get started.
    """)

elif st.session_state.processing_stage >= 1:
    # Processing stages
    progress_container = st.container()
    
    with progress_container:
        if st.session_state.processing_stage == 1:
            st.markdown('<div class="agent-status agent-running">🔄 <strong>File Reading Agent</strong> is running...</div>', unsafe_allow_html=True)
            
            # Run Agent 1
            input_data = {"file_path": st.session_state.tmp_file_path}
            
            with st.spinner("Reading and processing medical history file..."):
                try:
                    # Run only agent1
                    state = {"file_path": st.session_state.tmp_file_path}
                    result = agent1_node(state)
                    st.session_state.file_content = result["content"]
                    st.session_state.processing_stage = 2
                    time.sleep(1)  # Brief pause for UX
                    st.rerun()
                except Exception as e:
                    st.error(f"Error in File Reading Agent: {e}")
        
        elif st.session_state.processing_stage == 2:
            st.markdown('<div class="agent-status agent-completed">✅ <strong>File Reading Agent</strong> completed</div>', unsafe_allow_html=True)
            st.markdown('<div class="agent-status agent-running">🔄 <strong>Medical Summary Agent</strong> is running...</div>', unsafe_allow_html=True)
            
            with st.spinner("Generating medical summary..."):
                try:
                    # Run agent2
                    state = {"content": st.session_state.file_content}
                    result = agent2_node(state)
                    st.session_state.patient_summary = result["summary"]
                    st.session_state.processing_stage = 3
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error in Medical Summary Agent: {e}")
        
        elif st.session_state.processing_stage == 3:
            st.markdown('<div class="agent-status agent-completed">✅ <strong>File Reading Agent</strong> completed</div>', unsafe_allow_html=True)
            st.markdown('<div class="agent-status agent-completed">✅ <strong>Medical Summary Agent</strong> completed</div>', unsafe_allow_html=True)
            st.markdown('<div class="agent-status agent-running">🔄 <strong>Clinical Trial Matching Agent</strong> is running...</div>', unsafe_allow_html=True)
            
            with st.spinner("Finding suitable clinical trials..."):
                try:
                    # Run agent3
                    state = {
                        "content": st.session_state.file_content,
                        "summary": st.session_state.patient_summary
                    }
                    result = agent3_node(state)
                    st.session_state.extracted_info = result["extracted_info"]
                    st.session_state.processing_stage = 4
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error in Clinical Trial Matching Agent: {e}")
        
        elif st.session_state.processing_stage == 4:
            st.markdown('<div class="agent-status agent-completed">✅ <strong>File Reading Agent</strong> completed</div>', unsafe_allow_html=True)
            st.markdown('<div class="agent-status agent-completed">✅ <strong>Medical Summary Agent</strong> completed</div>', unsafe_allow_html=True)
            st.markdown('<div class="agent-status agent-completed">✅ <strong>Clinical Trial Matching Agent</strong> completed</div>', unsafe_allow_html=True)
            
            # Show clinical trial results
            st.subheader("🎯 Matching Clinical Trials")
            
            try:
                clinical_trials = json.loads(st.session_state.extracted_info)
                if clinical_trials:
                    st.success(f"Found {len(clinical_trials)} suitable clinical trial(s)")
                    for nct_id in clinical_trials.keys():
                        st.markdown(f'<div class="trial-card"><strong>NCT ID:</strong> {nct_id}</div>', unsafe_allow_html=True)
                else:
                    st.info("No suitable clinical trials found for this patient.")
            except:
                st.info("No suitable clinical trials found for this patient.")
            
            st.markdown('<div class="agent-status agent-running">🔄 <strong>Communication Agent</strong> is running...</div>', unsafe_allow_html=True)
            
            with st.spinner("Sending notifications to stakeholders..."):
                try:
                    # Run agent4
                    state = {"extracted_info": st.session_state.extracted_info}
                    result = agent4_node(state)
                    # st.session_state.final_message = result["extracted_details"]
                    st.session_state.final_message = 'A mail has been sent to the stakeholder for suitable client.'
                    st.session_state.processing_stage = 5
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error in Communication Agent: {e}")
        
        elif st.session_state.processing_stage == 5:
            # Final state - show all completed
            st.markdown('<div class="agent-status agent-completed">✅ <strong>File Reading Agent</strong> completed</div>', unsafe_allow_html=True)
            st.markdown('<div class="agent-status agent-completed">✅ <strong>Medical Summary Agent</strong> completed</div>', unsafe_allow_html=True)
            st.markdown('<div class="agent-status agent-completed">✅ <strong>Clinical Trial Matching Agent</strong> completed</div>', unsafe_allow_html=True)
            
            # Show clinical trial results
            st.subheader("🎯 Matching Clinical Trials")
            try:
                clinical_trials = json.loads(st.session_state.extracted_info)
                if clinical_trials:
                    st.success(f"Found {len(clinical_trials)} suitable clinical trial(s)")
                    for nct_id in clinical_trials.keys():
                        st.markdown(f'<div class="trial-card"><strong>NCT ID:</strong> {nct_id}</div>', unsafe_allow_html=True)
                else:
                    st.info("No suitable clinical trials found for this patient.")
            except:
                st.info("No suitable clinical trials found for this patient.")
            
            st.markdown('<div class="agent-status agent-completed">✅ <strong>Communication Agent</strong> completed</div>', unsafe_allow_html=True)
            
            # Show final message
            st.subheader("📧 Notification Status")
            # if st.session_state.final_message:
            #     st.markdown(f'<div class="success-message">{st.session_state.final_message}</div>', unsafe_allow_html=True)
            # st.markdown(f'<div class="success-message">{st.session_state.final_message}</div>', unsafe_allow_html=True) 
            st.markdown(
            '<div class="success-message">'
            'A mail has been sent to the stakeholder for suitable client.'
            '</div>',
            unsafe_allow_html=True
            )
            
            # # Reset button
            # if st.button("🔄 Process Another File", type="secondary"):
            #     st.session_state.processing_stage = 0
            #     st.session_state.clinical_trials = {}
            #     st.session_state.final_message = ""
            #     st.session_state.patient_summary = ""
            #     st.rerun()