# Building a Vector Search Engine with Langchain and Streamlit

## Find your ideal movie with an AI-powered vector search engine featuring +30k films

The project is a vector search engine designed to help you find your ideal movie based on plot description queries, allowing you to choose between `all-MiniLM-L6-v2` (open-source) and `text-embedding-3-small` (OpenAI) embedding models.

This search engine supports metadata filtering, enabling you to filter movies based on release dates. Additionally, the application manages the process of vectorizing content from a CSV dataset, saving textual data to Chroma DB.

## How to install 

1. Clone the project
2. Configure an OpenAI API Key if you don't have one and set it as an environment variable named `OPENAI_API_KEY` in a `.env` file within the directory
3. Download the [Wikipedia Movie Plots with AI Plot Summaries](https://www.kaggle.com/datasets/gabrieltardochi/wikipedia-movie-plots-with-plot-summaries) dataset from Kaggle and save the CSV file in a folder named `dataset`
4. Install the required libraries with `pip -r requirements.txt` (Python 3.10 or higher is recommended)
5. Vectorize the content and store the vectors in a Chroma database using the following command (approximately 15 minutes required)

```
>>> from app import load_csv_to_docs, \
                split_docs_to_chunks, \
                create_or_get_vectorstore
>>> documents = load_csv_to_docs()
>>> chunks = split_docs_to_chunks(documents)
>>> create_or_get_vectorstore('./dataset/wiki_movie_plots_deduped_with_summaries.csv','PlotSummary','OpenAI')
>>> create_or_get_vectorstore('./dataset/wiki_movie_plots_deduped_with_summaries.csv','PlotSummary','SentenceTransformer')
```

6. Run `streamlit run app.py` and open the application on localhost:8501.

![Vector Search Engine Demo](https://143998935.fs1.hubspotusercontent-eu1.net/hubfs/143998935/vector-search-engine-langchain-streamlit.gif)

## Customizing the project for your needs

This project serves as a good boilerplate template to kickstart the development of a vector search engine. You can customize the project with different data by providing an alternative CSV dataset. Simply make minor modifications by changing references to content and metadata columns while mostly keeping the code structure intact. Additionally, you can utilize alternative embedding models by modifying the `create_or_get_vectorstore` function.

## Like this project?

If you enjoyed this project, you may be interested in exploring more resources in Data Science, Engineering, and AI on [Datasense](https://www.datasense.space/)!