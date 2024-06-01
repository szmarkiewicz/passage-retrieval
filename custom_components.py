from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
import spacy
from spacy import Language
from typing import Any, Dict, List, Optional
from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace


@component
class PolishSpacyTextEmbedder(SentenceTransformersTextEmbedder):
    def __init__(self, model: str = "sentence-transformers/all-mpnet-base-v2", device: Optional[ComponentDevice] = None,
                 token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False), prefix: str = "",
                 suffix: str = "", batch_size: int = 32, progress_bar: bool = True, normalize_embeddings: bool = False,
                 trust_remote_code: bool = False):
        self.spacy_nlp = None
        self.model = model
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.trust_remote_code = trust_remote_code

    def warm_up(self):
        print("loading model")
        self.spacy_nlp = spacy.load(self.model)

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        if not isinstance(text, str):
            raise TypeError(
                "SentenceTransformersTextEmbedder expects a string as input."
                "In case you want to embed a list of Documents, please use the SentenceTransformersDocumentEmbedder."
            )
        if not hasattr(self, "spacy_nlp"):
            raise RuntimeError("The embedding model has not been loaded. Please call warm_up() before running.")

        text_to_embed = self.prefix + text + self.suffix
        tokens = self.spacy_nlp(text_to_embed)

        return {"embedding": tokens.vector.tolist()}


@component
class PolishSpacyDocumentEmbedder(SentenceTransformersDocumentEmbedder):
    def __init__(self, model: str = "sentence-transformers/all-mpnet-base-v2", device: Optional[ComponentDevice] = None,
                 token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False), prefix: str = "",
                 suffix: str = "", batch_size: int = 32, progress_bar: bool = True, normalize_embeddings: bool = False,
                 meta_fields_to_embed: Optional[List[str]] = None, embedding_separator: str = "\n",
                 trust_remote_code: bool = False):
        self.model = model
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.trust_remote_code = trust_remote_code
        self.spacy_nlp: Language or None = None

    def _get_telemetry_data(self) -> Dict[str, Any]:
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            model=self.model,
            device=self.device.to_dict(),
            token=self.token.to_dict() if self.token else None,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            trust_remote_code=self.trust_remote_code,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceTransformersDocumentEmbedder":
        init_params = data["init_parameters"]
        if init_params["device"] is not None:
            init_params["device"] = ComponentDevice.from_dict(init_params["device"])
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        return default_from_dict(cls, data)

    def warm_up(self):
        print("loading model")
        self.spacy_nlp = spacy.load(self.model)

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "SentenceTransformersDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a list of strings, please use the SentenceTransformersTextEmbedder."
            )
        if not hasattr(self, "spacy_nlp"):
            raise RuntimeError("The embedding model has not been loaded. Please call warm_up() before running.")

        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key]
            ]
            text_to_embed = (
                    self.prefix + self.embedding_separator.join(
                meta_values_to_embed + [doc.content or ""]) + self.suffix
            )
            texts_to_embed.append(text_to_embed)

        piped_texts = self.spacy_nlp.pipe(texts_to_embed)

        for doc, piped in zip(documents, piped_texts):
            print(piped)
            doc.embedding = piped.vector.tolist()

        return {"documents": documents}
