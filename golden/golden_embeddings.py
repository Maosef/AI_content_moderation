"""
Authors: John T. Halloran <halloranjt@leidos.com>
         Alexandra Keamy <alexandra.j.keamy@leidos.com>

"""
from typing import List, Final, Dict, Optional, Union, Any
import transformers
import torch
import numpy as np
from tqdm import tqdm
# from langchain.text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from golden.utils import (find_executable_batch_size, 
                          clear_torch_cache, 
                          should_reduce_batch_size, 
                          should_reduce_batch_size_but_handle_error)
import os
import logging
import sys
import torch.nn.functional as F

eval_logger = logging.getLogger("golden-embeddings")
eval_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
eval_logger.addHandler(handler)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = {"cuda" : ("cuda", "GPU detected") if torch.cuda.is_available() else ('cpu', "No GPU detected, defaulting to CPU"),
          "cpu" : ("cpu", "CPU selected"),
          }


# Per-embedding normalization specific functions, standarizes API calls
def m2_bert_norm(model_output, attention_mask = None): # "togethercomputer/m2-bert-80M-8k-retrieval"
    return model_output["sentence_embedding"]

# sentence-transformers/all-MiniLM-L6-v2
# See: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
def mean_pooling_and_norm(model_output, attention_mask): 
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return F.normalize(torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9), p=2, dim=1)

# thenlper/gte-large
# See: https://huggingface.co/thenlper/gte-large
def average_pool_and_norm(outputs: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden_states = outputs.last_hidden_state
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return F.normalize(last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None], p=2, dim=1)

# Alibaba-NLP/gte-Qwen2-1.5B-instruct
# See: https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct
def last_token_pool_and_norm(outputs: torch.Tensor,
                             attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden_states = outputs.last_hidden_state
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return F.normalize(last_hidden_states[:, -1],  p=2, dim=1)
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return F.normalize(last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths], p=2, dim=1)

MODEL_ZOO = {"togethercomputer/m2-bert-80M-8k-retrieval" : {"tokenizer_id" : "bert-base-uncased", 
                                                            "alias" : "m2-bert",
                                                            "norm" : m2_bert_norm,
                                                            },
             "sentence-transformers/all-MiniLM-L6-v2" : {"tokenizer_id" : "sentence-transformers/all-MiniLM-L6-v2",
                                                         "alias" : "all-MiniLM-L6-v2",
                                                         "norm" : mean_pooling_and_norm,
                                                         },
             "thenlper/gte-large" : {"tokenizer_id" : "thenlper/gte-large",
                                     "alias" : "gte-large",
                                     "norm" : average_pool_and_norm,
                                     },
             "Alibaba-NLP/gte-Qwen2-1.5B-instruct" : {"tokenizer_id" : "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                                     "alias" : "gte-Qwen2-1.5B-instruct",
                                     "norm" : last_token_pool_and_norm,
                                     },                                     
                                     }

MODEL_ID: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_ID: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH: Final[int] = 512
BATCH_SIZE: Final[int] = 32
NUM_WORKERS: Final[int] = 8
CHUNK_OVERLAP: Final[int] = 20
DICT_LANGUAGE_MAP: Final[Dict[str, Language]] = {l.name.lower(): l for l in Language}


# TODO: add support for multi-gpus via world-size, leveraging accelerate

def golden_embedding_options(kwargs: Dict[Any,Any] = {}):
    options = {"model_id" : MODEL_ID,
               "tokenizer_id" : TOKENIZER_ID,
               "max_seq_length" : MAX_SEQ_LENGTH,
               "device" : DEVICE["cuda"][0],
               "trust_remote_code" : True,
               "max_batch_size" : 64,
               "batch_size" : 1,
               "quantization_config" : None,
    }
    if kwargs:
        for k in kwargs:
            if k in options:
                options[k] = kwargs[k]
    print(kwargs)
    print(options)
    return options

def split_documents_given_language(texts: List[str],
                                   language: str = "",
                                   chunk_size = MAX_SEQ_LENGTH,
                                   chunk_overlap = CHUNK_OVERLAP)-> List[str]:
    if not language or language not in DICT_LANGUAGE_MAP:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                       chunk_overlap=chunk_overlap)
    else:
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language = DICT_LANGUAGE_MAP[language],
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap)        

    return [getattr(doc, "page_content", "") for doc in
            text_splitter.split_documents([Document(text) for text in texts])]


class Embedding(Embeddings):
    """ Given embedding model, associated tokenizer, and a corpus:
            - Embed each member of the corpus
            - Create FAISS index over all embeddings
            
        Given a set of queries and a specified top_k:
            - Return top_k closest db indices per query
    """
    def __init__(self,
                 model_id: str = MODEL_ID,
                 tokenizer_id: str = TOKENIZER_ID,
                 max_seq_length: int = MAX_SEQ_LENGTH,
                 device: str = 'cuda',
                 trust_remote_code: Optional[bool] = True,
                 max_batch_size: Optional[int] = 64,
                 batch_size: Optional[Union[int, str]] = "auto",
                 quantization_config = None,
                 ):
        self._world_size = 1
        self.trust_remote_code = trust_remote_code
        self._config = None
        self.AUTO_MODEL_CLASS = None
        self.max_batch_size = max_batch_size
        self.batch_size = batch_size
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        if device not in DEVICE:
            raise Exception(f"{device} specified, but must be one of 'cpu' or 'cuda'")
        device, msg = DEVICE[device]
        eval_logger.info(msg)
        self.device = torch.device(device)
        # # TODO: add metal support

        # Check if quantization file was supplied, then try to import BitsAndBytes...
        model_loaded = False
        if quantization_config:
            try:
                from transformers import BitsAndBytesConfig
                bandb = True
            except ImportError:
                bandb = False
            if bandb:
                if not type(quantization_config)==BitsAndBytesConfig:
                    eval_logger.info("Quantization config not of type BitsAndBytesConfig, loading full precision model")
                else:
                    if model_id == "togethercomputer/m2-bert-80M-8k-retrieval":
                        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
                            model_id, 
                            trust_remote_code=self.trust_remote_code,
                            quantization_config=quantization_config,
                            ).to(self.device)
                    else:
                        self.model = transformers.AutoModel.from_pretrained(
                            model_id, 
                            trust_remote_code=self.trust_remote_code,
                            quantization_config=quantization_config,
                            ).to(self.device) 
                    model_loaded = True
            else:
                eval_logger.info("Quantization config detected but BitsAndBytesConfig failed to import, loading full precision model")
        if not model_loaded:
            if model_id == "togethercomputer/m2-bert-80M-8k-retrieval":
                self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
                    model_id, 
                    trust_remote_code=self.trust_remote_code,
                ).to(self.device)
            else:
                self.model = transformers.AutoModel.from_pretrained(
                    model_id, 
                    trust_remote_code=self.trust_remote_code,
                ).to(self.device)                
        self.max_seq_length: int = max_seq_length
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_id,
                model_max_length=max_seq_length,
                )
        except Exception as Argument:
            raise Exception(f"Failed to load specified tokenizer {tokenizer_id} with error {Argument}")    

        if isinstance(batch_size, str) and "auto" in batch_size.lower():
            self.set_batch_size()
        else:
            eval_logger.info(f"Batch size set to {batch_size}")        

    @property
    def world_size(self):
        return self._world_size

    def _model_call(self, inps):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        embeddings returned from the model's forward
        """
        with torch.no_grad(): # Call forward
            return self.model(inps)
            # if self.model_id=="togethercomputer/m2-bert-80M-8k-retrieval"
            #     return self.model(inps)["sentence_embedding"]
            # else:
            #     return self.model(inps)


    def set_batch_size(self):
        eval_logger.info("Detecting largest batch size.")
        self.batch_size = self._detect_batch_size()
        eval_logger.info(f"Determined Largest batch size: {self.batch_size}")
        
    def embed_documents(self, 
                        texts: List[str],
                        num_workers = NUM_WORKERS,
                        device: str = 'cuda',
                        return_numpy: bool = False) -> Union[np.ndarray, List[List[float]]]:
        """ OpenAI Embeddings signature for reference: 
            embed_documents(texts: List[str], chunk_size: Optional[int] = 0) → List[List[float]]
        """
        embeddings =  np.array([])      
        for batch in tqdm(torch.utils.data.DataLoader(texts,
                                                      batch_size = self.batch_size,
                                                      shuffle=False,
                                                      num_workers=num_workers,
                                                      pin_memory=True,
                                                      batch_sampler=None,
                                                      sampler = None)):
            if not np.any(embeddings):
                embeddings = self.embed_batch(batch)
            else:
                embeddings = np.concatenate((embeddings, self.embed_batch(batch)), axis = 0)
        if return_numpy:
            return embeddings
        else:
            return embeddings.tolist()

    def embed_batch(self,
                    batch: List[str]) -> np.ndarray:
                    # device: str = '') -> np.ndarray:
        """ Make sure to transfer back to CPU
        """
        device = self.device
        with torch.no_grad():
            input_ids = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_seq_length
            ).to(device)
            # input_ids = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = self.model(**input_ids)
            if self.model_id in MODEL_ZOO:
                return MODEL_ZOO[self.model_id]["norm"](outputs, input_ids['attention_mask']).to('cpu').numpy()
            else:
                try:
                    return MODEL_ZOO[MODEL_ID]["norm"](outputs, input_ids['attention_mask']).to('cpu').numpy()
                except:
                    print(f"Could not find normalization function for {self.model_id}, tried {MODEL_ID} normalization but failed.  Exitting")
                    raise
            # if self.model_id=="togethercomputer/m2-bert-80M-8k-retrieval":
            #     return outputs['sentence_embedding'].to('cpu').numpy()
            # elif self.model_id=="sentence-transformers/all-MiniLM-L6-v2":
            #     sentence_embeddings = mean_pooling(outputs, input_ids['attention_mask'])
            #     sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            #     return sentence_embeddings.to("cpu").numpy()
            
        
    def embed_query(self, 
                    text: str,
                    device: str = '',
                    ) -> List[float]:
        return self.embed_batch(batch=[text]).squeeze().tolist()

    def _detect_batch_size(self, requests=None, pos: int = 0):
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_length = len(
                (context_enc + continuation_enc)[-(self.max_seq_length + 1) :][:-1]
            )
        else:
            max_length = self.max_seq_length

        def forward_batch(batch_size):
            call_kwargs = {}
            test_batch = torch.ones(
                (batch_size, max_length), device=self.device
            ).long()
            for _ in range(5):
                self._model_call(test_batch, **call_kwargs)  # noqa: F841

            return batch_size
        
        # def my_bisection(a,b):

        batch_size = self.max_batch_size
        bisect_left = 0
        bisect_right = batch_size        
        while True:
            # print(f"{batch_size}")
            clear_torch_cache()
            try:
                batch_size = forward_batch(batch_size)
            except RuntimeError as e:
                if should_reduce_batch_size(e): 
                    # temp = batch_size
                    # # past the optimum, bisect left
                    # batch_size = (batch_size + bisect_left)//2
                    # bisect_left = temp
                    # if batch_size==0:
                    #     return 1                    
                    batch_size //= 2
                elif should_reduce_batch_size_but_handle_error(e):
                    raise RuntimeError("Batch size {batch_size} caused nonrecoverable Cublas handle creation error, likely caused by excessively large batch size.  Try reducing the batch size a bit and trying again.")
                else:
                    clear_torch_cache()
                    return batch_size
                    # if bisect_left >= batch_size:
                    #     return batch_size
                    # else:
                    #     # some space, bisect right
                    #     bisect_left = (batch_size + bisect_left) // 2
            else:
                # clear_torch_cache()
                # if bisect_left >= batch_size:
                #     return batch_size
                # else:
                #     # some space, bisect right
                #     bisect_left = (batch_size + bisect_left) // 2                
                return batch_size

    def old_detect_batch_size(self, requests=None, pos: int = 0):
        """ Adapted from the Eleuther LM-Eval-Harness: 
        https://github.com/EleutherAI/lm-evaluation-harness/blob/7852985b2b5352df147067e01a121c52297f8821/lm_eval/models/huggingface.py

        Retrofitted for embeddings by John Halloran, halloranjt@leidos.com
        """
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_length = len(
                (context_enc + continuation_enc)[-(self.max_seq_length + 1) :][:-1]
            )
        else:
            max_length = self.max_seq_length

        # if OOM, then halves batch_size and tries again
        @find_executable_batch_size(starting_batch_size=self.max_batch_size)
        def forward_batch(batch_size):
            call_kwargs = {}
            test_batch = torch.ones(
                (batch_size, max_length), device=self.device
            ).long()
            for _ in range(5):
                self._model_call(test_batch, **call_kwargs)  # noqa: F841

            return batch_size

        try:
            batch_size = forward_batch()
        except RuntimeError as e:
            if "No executable batch size found" in str(e):
                batch_size = 1
            else:
                raise

        clear_torch_cache()
        return batch_size  