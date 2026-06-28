## Convert the documents format into standard object which can be used further in the pipeline!!
from pathlib import Path

from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


class DocumentLoader:

    def load(self, file_path: Path) -> list[Document]:
        """
        Load a document into LangChain Document objects.

        Parameters
        ----------
        file_path : Path
            Path to the uploaded document.

        Returns
        -------
        list[Document]
            List of LangChain Document objects.
        """

        extension = file_path.suffix.lower()

        if extension == ".pdf":
            loader = PyPDFLoader(str(file_path))

        elif extension == ".docx":
            loader = Docx2txtLoader(str(file_path))

        elif extension in [".txt", ".md"]:
            loader = TextLoader(
                str(file_path),
                encoding="utf-8"
            )

        else:
            raise ValueError(
                f"Unsupported document type: {extension}"
            )

        documents = loader.load()

        return documents