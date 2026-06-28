from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentSplitter:

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def split(self, documents: list[Document], doc_uuid: str) -> list[Document]:
        chunks = self.text_splitter.split_documents(documents)

        # Add chunk ids
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
            chunk.metadata["document_uuid"] = doc_uuid

        return chunks