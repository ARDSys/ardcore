from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ardcore.storage.file.utils import count_tokens, merge_punctuation_chunks


class ChunkingProtocol(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        pass


class TextGetterProtocol(ABC):
    @abstractmethod
    def get_text(self, data: dict) -> str:
        pass


class CleanTextGetter(TextGetterProtocol):
    def get_text(self, data: dict) -> str:
        return data["clean_full_text"]

    def __str__(self) -> str:
        return "CleanTextGetter"


class FullTextGetter(TextGetterProtocol):
    def get_text(self, data: dict) -> str:
        return data["full_text"]

    def __str__(self) -> str:
        return "FullTextGetter"


class SectionTextGetter(TextGetterProtocol):
    def get_text(self, data: dict) -> str:
        return data["sections"]

    def __str__(self) -> str:
        return "SectionTextGetter"


class CleanSectionTextGetter(TextGetterProtocol):
    def get_text(self, data: dict) -> str:
        return data["clean_sections"]

    def __str__(self) -> str:
        return "CleanSectionTextGetter"


@dataclass
class FixedChunking(ChunkingProtocol):
    chunk_size: int
    chunk_overlap: int
    length_function: Callable[[str], int] = count_tokens
    text_getter: TextGetterProtocol = FullTextGetter()

    def chunk(self, data: dict) -> list[str]:
        text = self.text_getter.get_text(data)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            is_separator_regex=False,
            separators=[". ", "? ", "! ", "\n\n", "\n", " "],
        )

        pages = splitter.split_text(text)
        # doesn't allow chunking in the middle of a sentence
        pages_merged = merge_punctuation_chunks(pages)

        return pages_merged

    def __str__(self) -> str:
        return f"FixedChunking(chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, text_getter={self.text_getter})"


class SectionChunking(ChunkingProtocol):
    text_getter: TextGetterProtocol = SectionTextGetter()

    def chunk(self, data: dict) -> list[str]:
        return self.text_getter.get_text(data)

    def __str__(self) -> str:
        return f"SectionChunking(text_getter={self.text_getter})"


class FullChunking(ChunkingProtocol):
    text_getter: TextGetterProtocol = FullTextGetter()

    def chunk(self, data: dict) -> list[str]:
        return [self.text_getter.get_text(data)]

    def __str__(self) -> str:
        return f"FullChunking(text_getter={self.text_getter})"
