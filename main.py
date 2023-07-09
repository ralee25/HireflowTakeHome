import os
# from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
# from langchain.chains import RetrievalQA
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.indexes import VectorstoreIndexCreator

os.environ["OPENAI_API_KEY"] = "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"

# working code
# video_id = "QsYGlZkevEg"
# loader = YoutubeLoader(video_id)
# docs=loader.load()

# index = VectorstoreIndexCreator()
# index = index.from_documents(docs)
# response = index.query("Where was the show the Last of Us filmed?")
# print(f"Answer: {response}")

# Two Karpathy lecture videos
urls = ["https://youtu.be/kCc8FmEb1nY", "https://youtu.be/VMj-3S1tku0"]

# Directory to save audio files
save_dir = "~/Downloads/YouTube"

# Transcribe the videos to text
loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser())
print(loader)
docs = loader.load()

YoutubeAudioLoader(urls, save_dir)

# Returns a list of Documents, which can be easily viewed or parsed
print(len(docs))