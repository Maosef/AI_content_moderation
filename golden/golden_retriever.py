from langchain_community.vectorstores.chroma import *

from golden.golden_embeddings import (Embedding,
                                      golden_embedding_options,
                                      MODEL_ID,
                                      TOKENIZER_ID,
                                      MAX_SEQ_LENGTH,
                                      BATCH_SIZE,
                                      NUM_WORKERS, 
                                      CHUNK_OVERLAP,
                                      DEVICE,
                                      split_documents_given_language,
                                      )
import chromadb
from typing import Final
import traceback
import os

import logging
import sys
import json

from golden.utils import clear_torch_cache, split_list

logger = logging.getLogger(__name__)
format='%(asctime)s %(message)s'
logging.basicConfig(filename="golden_retriever.log",
                    format=format,
                    filemode='w',
                    level=logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter(format))
logger.addHandler(handler)

SIMILARITY_KEY: Final[str] = "hnsw:space"
VALID_SIMILARITY_FNS: Final[set] = {"l2", "ip", "cosine"}
EMBEDDING_SETTING_FILE = "embedding_settings.json"

def set_similarity_fn(fn: Optional[str] = None, 
                      metadata: Optional[Dict] = None):
    if fn:
        fn = fn.lower()
        if fn in VALID_SIMILARITY_FNS:
            if metadata:
                metadata[SIMILARITY_KEY] = fn
            else:
                metadata = {
                    SIMILARITY_KEY: fn
                    }
        else:
            if metadata and SIMILARITY_KEY not in metadata:
                metadata[SIMILARITY_KEY] = "l2"
            elif not metadata:
                metadata = {
                    SIMILARITY_KEY: "l2"
                    }
    else:
        if metadata and SIMILARITY_KEY not in metadata:
            metadata[SIMILARITY_KEY] = "l2"
        elif not metadata:
            metadata = {
                SIMILARITY_KEY: "l2"
                }    

class Golden_Retriever(Chroma):
    """`ChromaDB` vector store.

    To use, you should have the ``chromadb`` python package installed.

    Example:
        .. code-block:: python

                from langchain_community.vectorstores import Chroma
                from langchain_community.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                vectorstore = Chroma("langchain_store", embeddings)
    """    
    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"        
    def __init__(
        self,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        collection_metadata: Optional[Dict] = None,
        client: Optional[chromadb.Client] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        similarity_fn: Optional[str] = None, # l2 = euclidean, cosine, ip = inner product
        **kwargs,
    ) -> None:
        
        if embedding_function is None:
            if "model_id" not in kwargs:
                kwargs["model_id"] = MODEL_ID
            if "tokenizer_id" not in kwargs:
                kwargs["tokenizer_id"] = TOKENIZER_ID
            if "max_seq_length" not in kwargs:
                kwargs["max_seq_length"] = MAX_SEQ_LENGTH
            embedding_function = Embedding(**golden_embedding_options(kwargs))
        else:
            for key in ["model_id", "tokenizer_id","max_seq_length", "max_batch_size", "batch_size"]:
                if hasattr(embedding_function, key):
                    kwargs[key]  = getattr(embedding_function, key)


        if persist_directory:
            self.write_embedding_settings_on_init(embedding_function, persist_directory)

        collection_metadata = set_similarity_fn(similarity_fn, 
                                                collection_metadata)

        super().__init__(collection_name,
                        embedding_function,
                        persist_directory,
                        client_settings,
                        collection_metadata,
                        client,
                        relevance_score_fn,
                        )
    @staticmethod
    def write_embedding_settings_on_init(embedding_function, persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
        output_file = os.path.join(persist_directory, EMBEDDING_SETTING_FILE)
        if embedding_function and isinstance(embedding_function, Embedding):
            if hasattr(embedding_function, 'model_id'):
                embedding_settings = {}
                embedding_settings["model_id"] = getattr(embedding_function, 'model_id')
                if hasattr(embedding_function, 'tokenizer_id'):
                    embedding_settings["tokenizer_id"] = getattr(embedding_function, 'tokenizer_id')
                embedding_settings["batch_size"] = getattr(embedding_function, 'batch_size', 1)
                embedding_settings["max_batch_size"] = getattr(embedding_function, 'max_batch_size', 1)
                # TODO: add support for quantization config
                logger.info(f"Saving embedding settings to {output_file}")
                print(embedding_settings)
                with open(output_file, "w") as f:
                    json.dump(embedding_settings, f)

    def clean_then_persist(self) -> None:
        """Persist the collection.

        This can be used to explicitly persist the data to disk.
        It will also be called automatically when the object is destroyed.

        Since Chroma 0.4.x the manual persistence method is no longer
        supported as docs are automatically persisted.
        """
        major, minor, _ = chromadb.__version__.split(".")
        if int(major) == 0 and int(minor) < 4:
            if self._persist_directory is None:
                raise ValueError(
                    "You must specify a persist_directory on"
                    "creation to persist the collection."
                )
            if os.path.isdir(self._persist_directory):
                try:
                    from shutil import rmtree
                    rmtree(self._persist_directory)
                except OSError as e:
                    print(f"Persist directory {self._persist_directory} could not be removed")
                    raise e
            if self._embedding_function is not None:
                self.write_embedding_settings_on_init(self._embedding_function, self._persist_directory)
            self._client.persist()
        else:
            print("Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.")

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        num_workers = NUM_WORKERS,
        device: str = 'cuda',
        **kwargs,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        # TODO: align this with from_texts and from_documents
        embeddings = None
        texts = list(texts)
        if self._embedding_function is not None:         
            if type(self._embedding_function)==Embedding:
                embeddings = self._embedding_function.embed_documents(texts = texts,
                                                                      num_workers = num_workers,
                                                                      device = device,
                                                                      )
            else:
                embeddings = self._embedding_function.embed_documents(texts)
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        if ids is None or len(ids) != len(texts):
            ids = [str(uuid.uuid1()) for _ in texts]
        if metadatas:
            # fill metadatas with empty dicts if somebody
            # did not specify metadata for all texts
            length_diff = len(texts) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff
            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)
            if non_empty_ids:
                metadatas = [metadatas[idx] for idx in non_empty_ids]
                texts_with_metadatas = [texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = (
                    [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    self._collection.upsert(
                        metadatas=metadatas,
                        embeddings=embeddings_with_metadatas,
                        documents=texts_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata from the document using "
                            "langchain_community.vectorstores.utils.filter_complex_metadata."
                        )
                        raise ValueError(e.args[0] + "\n\n" + msg)
                    else:
                        raise e
            if empty_ids:
                texts_without_metadatas = [texts[j] for j in empty_ids]
                embeddings_without_metadatas = (
                    [embeddings[j] for j in empty_ids] if embeddings else None
                )
                ids_without_metadatas = [ids[j] for j in empty_ids]
                self._collection.upsert(
                    embeddings=embeddings_without_metadatas,
                    documents=texts_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:
            self._collection.upsert(
                embeddings=embeddings,
                documents=texts,
                ids=ids,
            )
        return ids
    
    def update_documents(self, 
                         ids: List[str], 
                         documents: List[Document],
                         language: str = "",
                         chunk_size: Optional[int] = MAX_SEQ_LENGTH,
                         chunk_overlap: Optional[int] = CHUNK_OVERLAP,
                         num_workers = NUM_WORKERS,
                         device: str = 'cuda',
                         **kwargs,
                         ) -> None:
        """Update a document in the collection.

        Args:
            ids (List[str]): List of ids of the document to update.
            documents (List[Document]): List of documents to update.
        """
        # Do doc level processing?
        do_chunking: str = kwargs.pop("do_chunking", True)

        text = [document.page_content for document in documents]
        metadata = [document.metadata for document in documents]
        if self._embedding_function is None:
            raise ValueError(
                "For update, you must specify an embedding function on creation."
            )
        if type(self._embedding_function)==Embedding:
            embeddings, text = self._embedding_function.embed_documents(text,
                                                                        language = language,
                                                                        chunk_size = chunk_size,
                                                                        chunk_overlap = chunk_overlap,
                                                                        num_workers = num_workers,
                                                                        device = device,
                                                                        do_chunking = do_chunking,
                                                                        )
            
        else:
            embeddings = self._embedding_function.embed_documents(text)

        if hasattr(
            self._collection._client, "max_batch_size"
        ):  # for Chroma 0.4.10 and above
            from chromadb.utils.batch_utils import create_batches

            for batch in create_batches(
                api=self._collection._client,
                ids=ids,
                metadatas=metadata,
                documents=text,
                embeddings=embeddings,
            ):
                self._collection.update(
                    ids=batch[0],
                    embeddings=batch[1],
                    documents=batch[3],
                    metadatas=batch[2],
                )
        else:
            self._collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=text,
                metadatas=metadata,
            )

    @classmethod
    def load(cls,
            folder_path: str,
            **kwargs: Any,
            ):
        # TODO: store embedding info
        similarity_fn = kwargs.pop("similarity_fn", "cosine") # Otherwise, use ChromaDB/Golden Retriever

        embedding = kwargs.pop("embedding", None)
        embedding_settings_file = os.path.join(folder_path, EMBEDDING_SETTING_FILE)
        if os.path.isfile(embedding_settings_file):
            logger.info(f"Loading embedding settings from {embedding_settings_file}")
            with open(embedding_settings_file, "r") as json_file:
                embedding_settings = json.load(json_file)
                print(embedding_settings)
                if embedding_settings["model_id"] and embedding_settings["tokenizer_id"]:
                    embedding = Embedding(**embedding_settings)
        return cls(persist_directory = folder_path,
                   embedding_function = embedding,
                   similarity_fn = similarity_fn)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.Client] = None,
        collection_metadata: Optional[Dict] = None,
        chunk_size: Optional[int] = MAX_SEQ_LENGTH,
        chunk_overlap: Optional[int] = CHUNK_OVERLAP,
        max_embedding_buffer: int = 500000, # max number of embeddings to generate before adding to index
        **kwargs: Any,
    ):
        """Create a Chroma vectorstore from a raw documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            texts (List[str]): List of texts to add to the collection.
            collection_name (str): Name of the collection to create.
            persist_directory (Optional[str]): Directory to persist the collection.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            client_settings (Optional[chromadb.config.Settings]): Chroma client settings
            collection_metadata (Optional[Dict]): Collection configurations.
                                                  Defaults to None.

        Returns:
            Chroma: Chroma vectorstore.
        """

        # TODO: Add support for MetaData partitioning given chunking

        # batch_size: Union[str, int] = kwargs.pop("batch_size", "auto")
        # max_batch_size: int = kwargs.pop("max_batch_size", 512)
        # max_seq_length: int = kwargs.pop("max_seq_length", MAX_SEQ_LENGTH)
        # similarity_fn: str = kwargs.pop("similarity_fn", "cosine")
        # Do doc level processing?
        do_chunking: str = kwargs.pop("do_chunking", True)
        
        num_workers: int = kwargs.pop("num_workers", NUM_WORKERS)
        language: str = kwargs.pop("language", "")
        device: str = kwargs.pop("device", DEVICE["cuda"][0])

        golden_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            **kwargs,
        )

        if len(texts) <= max_embedding_buffer:
            if do_chunking:
                logger.info(f"Chunking {len(texts)} documents")
                texts = split_documents_given_language(texts,
                                                       language = language,
                                                       chunk_size = chunk_size,
                                                       chunk_overlap = chunk_overlap)
                ids = [str(uuid.uuid1()) for _ in texts]
                logger.info(f"Produced {len(texts)} chunks")
            else:
                logger.info(f"No chunking specified")
            if not ids:
                ids = [str(uuid.uuid1()) for _ in texts]
            golden_collection.add_texts(
                texts=texts,
                metadatas=None,
                ids=ids,
                num_workers=num_workers,
                device = device,
                )
        else:
            # NOOPUR: fix for bug
            # UnboundLocalError: local variable 'ids' referenced before assignment
            partition = 0
            num_texts = 0
            num_split_texts = 0
            for _texts in split_list(texts, max_embedding_buffer):
                _metadatas = None
                num_texts += len(_texts)

                if do_chunking:
                    logger.info(f"Chunking {len(_texts)} documents")
                    _texts = split_documents_given_language(_texts,
                                                            language = language,
                                                            chunk_size = chunk_size,
                                                            chunk_overlap = chunk_overlap)
                    logger.info(f"Produced {len(_texts)} chunks")
                else:
                    logger.info(f"No chunking specified")
                local_ids = [str(uuid.uuid1()) for _ in _texts]

                num_split_texts += len(_texts)
                logger.info(f"Evaluating text partition {partition}")
                try:
                    golden_collection.add_texts(
                        texts=_texts,
                        metadatas=_metadatas,
                        ids=local_ids,
                        num_workers=num_workers,
                        device = device,
                        )
                except Exception as e:
                    logger.info(f"Partiton {partition}: Processed {num_texts} raw documents, {num_split_texts} chunks")
                    logger.info(f"Encountered exception {e}, saving progress and exiting")
                    golden_collection.persist()
                    logger.debug(traceback.format_exc())

                del _texts, local_ids
                clear_torch_cache()
                partition += 1
            logger.info(f"Golden Retriever completed: processed {partition} partitions, {num_texts} raw documents, {num_split_texts} chunks")
            # for _texts in split_list(texts, max_embedding_buffer):
            #     # TODO: move metadata to match split and chunking
            #     _metadatas = None
            #     num_texts += len(_texts)

            #     if do_chunking:
            #         logger.info(f"Chunking {len(_texts)} documents")
            #         _texts = split_documents_given_language(_texts,
            #                                                 language = language,
            #                                                 chunk_size = chunk_size,
            #                                                 chunk_overlap = chunk_overlap)
            #         ids = [str(uuid.uuid1()) for _ in _texts]
            #         logger.info(f"Produced {len(_texts)} chunks")
            #     else:
            #         logger.info(f"No chunking specified")
            #     if not ids:
            #         ids = [str(uuid.uuid1()) for _ in _texts]                 

            #     num_split_texts += len(_texts)
            #     logger.info(f"Evaluating text partition {partition}")
            #     try:
            #         golden_collection.add_texts(
            #             texts=_texts,
            #             metadatas=_metadatas,
            #             ids=ids,
            #             num_workers=num_workers,
            #             device = device,
            #             )                    
            #     except Exception as e:
            #         logger.info(f"Partiton {partition}: Processed {num_texts} raw documents, {num_split_texts} chunks")
            #         logger.info(f"Encountered exception {e}, saving progress and exiting")
            #         golden_collection.persist()
            #         logger.debug(traceback.format_exc())

            #     del _texts, ids
            #     clear_torch_cache()
            #     partition += 1
            # logger.info(f"Golden Retriever completed: processed {partition} partitions, {num_texts} raw documents, {num_split_texts} chunks")

        return golden_collection     

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.Client] = None,  # Add this line
        collection_metadata: Optional[Dict] = None,
        similarity_fn: Optional[str] = None, # l2 = euclidean, cosine, ip = inner product
        **kwargs: Any,
    ):
        collection_metadata = set_similarity_fn(similarity_fn, 
                                                collection_metadata)
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        # if not embedding:
        #     embedding = Embedding(**golden_embedding_options(kwargs))
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            **kwargs,
        )