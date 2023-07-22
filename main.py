import os

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# OPENAI Key
OPENAI_API_KEY = input("Input API key for OpenAI: ")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Example YouTube urls
# Two Karpathy lecture videos (longer)
# urls = ["https://youtu.be/kCc8FmEb1nY", "https://youtu.be/VMj-3S1tku0"]

# Pedro Pascal video (shorter)
# urls = ["https://www.youtube.com/watch?v=QsYGlZkevEg&t=52s&ab_channel=SaturdayNightLive"]

# Barbie w/puppies
# urls = ["https://www.youtube.com/watch?v=s5bI_732Cqs&t=19s&ab_channel=BuzzFeedCeleb"]
#Collecting the YouTube Urls we want ChatBot to learn from
urls=[]
while True:
    youtube_url = input("Insert YouTube url you wish ChatBot to learn from (at least 1 | press 'q' or 'quit' to quit adding): ")
    if (youtube_url.lower() != "q" and youtube_url.lower() != "quit"):
        urls.append(youtube_url)
    else:
        # must add at least one url so keeps prompting user same question if they immediately quit without adding 
        if (len(urls) > 0 and (youtube_url.lower() == "q" or youtube_url.lower() == "quit")):
            break

# print(urls)

# Change directory as needed
save_dir = "/Users/rachel/Downloads/YouTube"

# Transcribe the videos to text
loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser())
docs = loader.load()

# debugging
# works yay!
# print(docs[0].page_content[0:100])

# Text embedding time!
# Combine doc(s)
combined_docs = [doc.page_content for doc in docs]
text = " ".join(combined_docs)

# Split the docs with Recurisve Char Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap = 0)
splits = text_splitter.split_text(text)
# Index building and vector stores!
embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_texts(splits, embeddings)


# Getting input from user 

# First build the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
)

# Testing!
# query="What did Pedro Pascal film most recently?"
# print(qa_chain.run(query))
# Answers: "Pedro Pascal most recently filmed a show called The Last of Us on HBO."

# Actual Q/A Session
while True:
    query = input("What's your question? (Press 'q' or 'quit' to exit the program.) ")
    if (query.lower() == "q" or query.lower() == "quit"):
        print("Thanks for using this imitation ChatBot!")
        break
    else:
        response = qa_chain.run(query)
        print(f"Answer: {response}")
        # print(qa_chain.run(query))

