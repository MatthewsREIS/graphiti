"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import json
import logging
from collections.abc import Iterable

from ..utils.bedrock import BedrockCredentials, create_bedrock_client
from .client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = 'amazon.titan-embed-text-v2:0'
COHERE_MAX_BATCH = 96


class AmazonBedrockEmbedderConfig(EmbedderConfig):
    model: str = DEFAULT_EMBEDDING_MODEL
    region: str = 'us-east-1'
    credentials: BedrockCredentials | None = None


class AmazonBedrockEmbedder(EmbedderClient):
    """Bedrock embedder for Amazon Titan (v1/v2) and Cohere Embed (v3/v4) models."""

    def __init__(self, config: AmazonBedrockEmbedderConfig | None = None):
        self.config = config or AmazonBedrockEmbedderConfig()
        self.client = create_bedrock_client(
            'bedrock-runtime', self.config.region, self.config.credentials
        )

    @property
    def _is_cohere(self) -> bool:
        return 'cohere' in self.config.model

    def _invoke(self, body: dict) -> dict:
        response = self.client.invoke_model(
            modelId=self.config.model,
            body=json.dumps(body),
            accept='application/json',
            contentType='application/json',
        )
        return json.loads(response['body'].read().decode('utf-8'))

    def _embed_titan(self, text: str) -> list[float]:
        body: dict = {'inputText': text}
        if 'titan-embed-text-v2' in self.config.model:
            body['dimensions'] = self.config.embedding_dim
            body['normalize'] = True
        return self._invoke(body)['embedding']

    def _embed_cohere(self, texts: list[str]) -> list[list[float]]:
        body = {
            'texts': texts,
            'input_type': 'search_document',
            'embedding_types': ['float'],
        }
        if 'embed-v4' in self.config.model:
            body['output_dimension'] = self.config.embedding_dim
        embeddings = self._invoke(body)['embeddings']
        # v3 returns a list of vectors; with embedding_types it returns {'float': [...]}
        return embeddings['float'] if isinstance(embeddings, dict) else embeddings

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, list):
            text = ' '.join(str(item) for item in input_data)
        else:
            text = str(input_data)

        try:
            loop = asyncio.get_running_loop()
            if self._is_cohere:
                return (await loop.run_in_executor(None, self._embed_cohere, [text]))[0]
            return await loop.run_in_executor(None, self._embed_titan, text)
        except Exception as e:
            logger.error(f'Bedrock embedding failed: {e}')
            raise

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        if not self._is_cohere:
            # Titan takes one input per call
            return list(await asyncio.gather(*(self.create(text) for text in input_data_list)))

        loop = asyncio.get_running_loop()
        embeddings: list[list[float]] = []
        try:
            for i in range(0, len(input_data_list), COHERE_MAX_BATCH):
                chunk = input_data_list[i : i + COHERE_MAX_BATCH]
                embeddings.extend(await loop.run_in_executor(None, self._embed_cohere, chunk))
        except Exception as e:
            logger.error(f'Bedrock embedding failed: {e}')
            raise
        return embeddings
