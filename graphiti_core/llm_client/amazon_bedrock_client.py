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
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..prompts.models import Message
from ..utils.bedrock import BedrockCredentials, create_bedrock_client
from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError

if TYPE_CHECKING:
    from botocore.exceptions import ClientError
else:
    from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

THROTTLING_CODES = {
    'ThrottlingException',
    'TooManyRequestsException',
    'ServiceQuotaExceededException',
    'ModelNotReadyException',
}

DEFAULT_MODEL = 'openai.gpt-oss-120b-1:0'


class AmazonBedrockLLMClient(LLMClient):
    """LLM client for any Bedrock chat model (gpt-oss, Claude, Nova, Llama, ...).

    Uses the model-agnostic Converse API. Structured output relies on the JSON schema that
    LLMClient.generate_response appends to the prompt, the same approach as the Groq client.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        max_tokens: int | None = None,
        region: str = 'us-east-1',
        credentials: BedrockCredentials | None = None,
    ):
        super().__init__(config, cache=False)

        self.region = region
        self.model = config.model if config and config.model else DEFAULT_MODEL
        self.small_model = config.small_model if config and config.small_model else self.model
        self.temperature = config.temperature if config and config.temperature is not None else 0.7
        if max_tokens is not None:
            self.max_tokens = max_tokens

        self.client = create_bedrock_client('bedrock-runtime', self.region, credentials)

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
        *,
        model: str | None = None,
    ) -> dict:
        # Schema injection is handled by LLMClient.generate_response - do not re-inject here.
        selected_model = model or (
            self.small_model if model_size == ModelSize.small else self.model
        )

        text = await self._converse(
            model=selected_model,
            messages=[msg.model_dump() for msg in messages],
            temperature=self.temperature,
            max_tokens=max_tokens,
            response_format='json' if response_model else 'text',
        )

        if not response_model:
            return {'content': text}

        try:
            return response_model.model_validate_json(text).model_dump()
        except Exception as e:
            # Never log the response or the validation input: it can contain episode data.
            logger.error(
                f'Failed to parse structured Bedrock response ({type(e).__name__}, {len(text)} chars)'
            )
            raise

    async def _converse(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        response_format: str,
    ) -> str:
        system = [{'text': m['content']} for m in messages if m['role'] == 'system']
        converse_messages = [
            {'role': m['role'], 'content': [{'text': m['content']}]}
            for m in messages
            if m['role'] != 'system'
        ]

        request: dict[str, Any] = {
            'modelId': model,
            'messages': converse_messages,
            'inferenceConfig': {'maxTokens': max_tokens, 'temperature': temperature},
        }
        if system:
            request['system'] = system

        try:
            # boto3 is sync: run it in the default executor
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(None, lambda: self.client.converse(**request))
        except ClientError as e:
            code = e.response.get('Error', {}).get('Code', '')
            if code in THROTTLING_CODES:
                raise RateLimitError(f'Rate limit exceeded: {code}') from e
            logger.error(f'Bedrock model invocation failed: {code or type(e).__name__}')
            raise
        except Exception as e:
            logger.error(f'Bedrock model invocation failed: {type(e).__name__}')
            raise

        # Reasoning models (gpt-oss) return reasoningContent blocks alongside text: keep text only.
        blocks = resp.get('output', {}).get('message', {}).get('content', [])
        text = ''.join(b['text'] for b in blocks if 'text' in b)

        if resp.get('stopReason') == 'max_tokens':
            logger.warning(f'Bedrock response from {model} was truncated at {max_tokens} tokens')

        if response_format == 'json':
            text = self._clean_json_response(text)
        return text.strip()

    def _clean_json_response(self, text: str) -> str:
        """Extract the JSON object from a model response (markdown fences, surrounding prose)."""
        text = text.strip()
        text = re.sub(r'^```(?:json|JSON)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)

        start = text.find('{')
        if start == -1:
            return text.strip()
        candidate = text[start:]

        # Decode normally first. Doubled braces ({{...}}) are a template artifact some models
        # echo back, so recover from them only when a normal decode fails.
        decoder = json.JSONDecoder()
        for attempt in (candidate, re.sub(r'\}\}\s*$', '}', candidate[1:])):
            try:
                _, end = decoder.raw_decode(attempt)
                return attempt[:end].strip()
            except ValueError:
                continue
        return candidate.strip()
