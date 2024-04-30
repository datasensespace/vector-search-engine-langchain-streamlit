import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

def load_csv_to_docs(file_path:str="./dataset/wiki_movie_plots_deduped_with_summaries.csv", 
                     content_col:str="PlotSummary"
                     ) -> list:
    """
    Load a CSV file into documents using Langchain DataFrame loader.

    Args:
        file_path (str): The file path to the CSV file.
        content_col (str): The name of the column containing the content of each document.

    Returns:
        list: A list of documents loaded from the CSV file.
    """

    df = pd.read_csv(file_path)

    loader = DataFrameLoader(df, page_content_column=content_col)

    documents = loader.load()

    return documents

def split_docs_to_chunks(documents:list, chunk_size:int=1000, chunk_overlap:int=0) -> list:
    """
    Split documents into chunks and format each chunk.

    Args:
        documents (list): A list of documents to be split.
        chunk_size (int, optional): The size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The overlap between consecutive chunks. Defaults to 0.

    Returns:
        list: A list of formatted chunks.
    """
    # Create a RecursiveCharacterTextSplitter instance
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Split documents into chunks using the text splitter
    chunks = text_splitter.split_documents(documents)
    
    # Iterate over each chunk
    for chunk in chunks:
        # Extract metadata from the chunk
        title = chunk.metadata['Title']
        origin = chunk.metadata['Origin/Ethnicity']
        genre = chunk.metadata['Genre']
        release_year = chunk.metadata['Release Year']
        
        # Extract content from the chunk
        content = chunk.page_content
        
        # Format the content with metadata
        final_content = f"TITLE: {title}\nORIGIN: {origin}\nGENRE: {genre}\nYEAR: {release_year}\nBODY: {content}\n"
        
        # Update the page content of the chunk with formatted content
        chunk.page_content = final_content
    
    return chunks

def create_or_get_vectorstore(file_path: str, content_col: str, selected_embedding: str) -> Chroma:
    """
    Create or get a Chroma vector store based on the selected embedding model.

    Args:
        file_path (str): The file path to the dataset.
        content_col (str): The name of the column containing the content of each document.
        selected_embedding (str): The selected embedding model ('OpenAI' or 'SentenceTransformer').

    Returns:
        Chroma: A Chroma vector store.
    """
    # Determine the embedding function and database path based on the selected embedding model
    if selected_embedding == 'OpenAI':
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=100, show_progress_bar=True)
        db_path = './chroma_openai'

    elif selected_embedding == 'SentenceTransformer':
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db_path = './chroma_hf'

    # Check if the database directory exists
    if not os.path.exists(db_path):
        # If the directory does not exist, create the database
        documents = load_csv_to_docs(file_path, content_col)
        chunks = split_docs_to_chunks(documents)

        print("CREATING DB...")
        db = Chroma.from_documents(chunks, embedding_function, persist_directory=db_path)

    else:
        # If the directory exists, load the existing database
        print('LOADING DB...')
        db = Chroma(persist_directory=db_path, embedding_function=embedding_function)

    return db

def query_vectorstore(db:Chroma, 
                      query:str, 
                      k:int=20, 
                      filter_dict:dict={}
                      ) -> pd.DataFrame:
    """
    Query a Chroma vector store for similar documents based on a query.

    Args:
        db (Chroma): The Chroma vector store to query.
        query (str): The query string.
        k (int, optional): The number of similar documents to retrieve. Defaults to 20.
        filter_dict (dict, optional): A dictionary specifying additional filters. Defaults to {}.

    Returns:
        pd.DataFrame: A DataFrame containing metadata of the similar documents.
    """
    # Perform similarity search on the vector store
    results = db.similarity_search(query, filter=filter_dict, k=k)

    # Initialize an empty list to store metadata from search results
    results_metadata = []

    # Extract metadata from results
    for doc in results:
        results_metadata.append(doc.metadata)

    # Convert metadata to a DataFrame
    df = pd.DataFrame(results_metadata)
    
    # Drop duplicate rows based on the 'Wiki Page' column
    df.drop_duplicates(subset=['Wiki Page'], keep='first', inplace=True)

    return df

def main():
    
    # Apply Streamlit page config
    st.set_page_config(
    page_title=" Vector Search Engine | Datasense",
    page_icon="https://143998935.fs1.hubspotusercontent-eu1.net/hubfs/143998935/Datasense_Favicon-2.svg"
    )

    # Read and apply custom CSS style
    with open('./css/style.css') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    # Display logo and title
    st.image("https://143998935.fs1.hubspotusercontent-eu1.net/hubfs/143998935/Datasense%20Logo_White.svg", width=180)
    st.title("Building a Vector Search Engine with Langchain and Streamlit")
    st.markdown("Find your ideal movie among over 30k films with an AI-powered vector search engine.")

    # Toggle for using OpenAI embeddings
    if "openai_on" not in st.session_state:
        st.session_state.openai_on = False

    openai_on = st.toggle('Use OpenAI embeddings')

    # Check if the toggle value has changed
    if openai_on != st.session_state.openai_on:
        # Clear the existing database from the session state
        if "db" in st.session_state:
            del st.session_state.db

    # Determine selected embedding model
    if openai_on:
        selected_embedding = "OpenAI"
        st.session_state.openai_on = True
    else:
        selected_embedding = "SentenceTransformer"
        st.session_state.openai_on = False

    # Create or get the vector store database
    file_path = './dataset/wiki_movie_plots_deduped_with_summaries.csv'
    content_col = 'PlotSummary'
    st.session_state.db = create_or_get_vectorstore(file_path, content_col, selected_embedding)

    # Text input for query
    query = st.text_input("Tonight I'd like to watch...", "A thriller movie about memory loss")

    # Display filter options
    filter_box = st.empty()
    with st.expander("Filter by year"):
        with st.form("filters"):
            filter_dict = {}
            st.write("Release year...")
            year_operator = st.selectbox(
                label="Operator",
                options=("is equal to", "is after", "is before")
            )
            year = st.number_input(
                label="Year",
                min_value=1900,
                max_value=2023,
                value=2000
            )
            submitted = st.form_submit_button("Apply filter")
            operator_signs = {
                "is equal to": "$eq",
                "is after": "$gt",
                "is before": "$lt"
            }

            if submitted:
                filter_dict = {
                    "Release Year": {
                        f"{operator_signs[year_operator]}": year
                    }
                }
                filter_box.markdown(
                    f"<p><b>Active filter</b>:</p> <span class='active-filter'>Released year {year_operator} {year}</span>", 
                    unsafe_allow_html=True
                )

    # Perform search if query exists
    if query:
        # Perform vector store query
        results_df = query_vectorstore(
            db=st.session_state.db,
            query=query,
            filter_dict=filter_dict
        )

        # Display search results
        for index, row in results_df.iterrows():
            st.markdown(
                f"""
                <div class='result-item-box'>
                    <span class='label-genre'>{row['Genre']}</span>
                    <h4>{row['Title']}</h4>
                    <div class='metadata'>
                        <p><b>Year:</b> {row['Release Year']}</p>
                        <p><b>Director:</b> {row['Director']}</p>
                        <p><b>Origin:</b> {row['Origin/Ethnicity']}</p>
                    </div>
                    <a href='{row['Wiki Page']}'>Read more →</a>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == '__main__':
    main()