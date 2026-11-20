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

from ..utils.bedrock import BedrockCredentials, create_bedrock_client, error_label
from .client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = 'amazon.titan-embed-text-v2:0'
TITAN_V2_DIMENSIONS = (256, 512, 1024)


class AmazonBedrockEmbedderConfig(EmbedderConfig):
    model: str = DEFAULT_EMBEDDING_MODEL
    region: str = 'us-east-1'
    credentials: BedrockCredentials | None = None


class AmazonBedrockEmbedder(EmbedderClient):
    """Bedrock embedder for Amazon Titan Text Embeddings (v1 and v2).

    Cohere Embed is not supported: it needs to know whether text is a search query or a stored
    document, and Graphiti's embedder interface does not carry that.
    """

    def __init__(self, config: AmazonBedrockEmbedderConfig | None = None):
        self.config = config or AmazonBedrockEmbedderConfig()

        if 'titan-embed' not in self.config.model:
            raise ValueError(
                f'Unsupported Bedrock embedding model {self.config.model!r}: only Amazon Titan '
                'embedding models are supported'
            )
        if (
            'titan-embed-text-v2' in self.config.model
            and self.config.embedding_dim not in TITAN_V2_DIMENSIONS
        ):
            raise ValueError(
                f'Titan Text Embeddings v2 supports 256, 512 or 1024 dimensions, '
                f'not {self.config.embedding_dim}'
            )

        self.client = create_bedrock_client(
            'bedrock-runtime', self.config.region, self.config.credentials
        )

    def _embed(self, text: str) -> list[float]:
        body: dict = {'inputText': text}
        if 'titan-embed-text-v2' in self.config.model:
            body['dimensions'] = self.config.embedding_dim
            body['normalize'] = True
        response = self.client.invoke_model(
            modelId=self.config.model,
            body=json.dumps(body),
            accept='application/json',
            contentType='application/json',
        )
        return json.loads(response['body'].read().decode('utf-8'))['embedding']

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
            return await loop.run_in_executor(None, self._embed, text)
        except Exception as e:
            # Never log the message: it can echo the text being embedded.
            logger.error(f'Bedrock embedding failed: {error_label(e)}')
            raise

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        # Titan takes one input per call
        return list(await asyncio.gather(*(self.create(text) for text in input_data_list)))
