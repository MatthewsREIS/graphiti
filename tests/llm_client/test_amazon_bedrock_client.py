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

# Running tests: pytest -xvs tests/llm_client/test_amazon_bedrock_client.py

import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from pydantic import BaseModel

from graphiti_core.llm_client.amazon_bedrock_client import AmazonBedrockLLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.prompts.models import Message
from graphiti_core.utils.bedrock import BedrockCredentials


class ResponseModel(BaseModel):
    """Test model for response testing."""

    test_field: str
    optional_field: int = 0


@pytest.fixture(autouse=True)
def mock_boto3_client():
    """Fixture to mock the boto3 bedrock-runtime client."""
    with patch('boto3.client') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def bedrock_client(mock_boto3_client):
    """Fixture to create an AmazonBedrockLLMClient with a mocked boto3 client."""
    config = LLMConfig(
        model='us.anthropic.claude-sonnet-4-20250514-v1:0', temperature=0.5, max_tokens=1000
    )
    client = AmazonBedrockLLMClient(config=config, region='us-east-1')
    client.client = mock_boto3_client
    return client


def converse_response(text: str, stop_reason: str = 'end_turn', reasoning: str | None = None):
    content = []
    if reasoning:
        content.append({'reasoningContent': {'reasoningText': {'text': reasoning}}})
    content.append({'text': text})
    return {
        'output': {'message': {'role': 'assistant', 'content': content}},
        'stopReason': stop_reason,
    }


MESSAGES = [
    Message(role='system', content='System message'),
    Message(role='user', content='User message'),
]


class TestAmazonBedrockLLMClientInitialization:
    def test_init_with_config(self):
        config = LLMConfig(model='test-model', temperature=0.5, max_tokens=1000)
        client = AmazonBedrockLLMClient(config=config, max_tokens=1000, region='eu-west-1')

        assert client.model == 'test-model'
        assert client.small_model == 'test-model'
        assert client.temperature == 0.5
        assert client.max_tokens == 1000
        assert client.region == 'eu-west-1'

    def test_init_without_config_defaults_to_gpt_oss(self):
        client = AmazonBedrockLLMClient(region='ap-southeast-1')

        assert client.model == 'openai.gpt-oss-120b-1:0'
        assert client.temperature == 0.7
        assert client.region == 'ap-southeast-1'

    def test_init_default_region(self):
        assert AmazonBedrockLLMClient().region == 'us-east-1'

    def test_credentials_are_passed_to_boto3(self):
        creds = BedrockCredentials(access_key_id='AKIA', secret_access_key='secret')
        with patch('graphiti_core.llm_client.amazon_bedrock_client.create_bedrock_client') as m:
            AmazonBedrockLLMClient(region='us-west-2', credentials=creds)
        m.assert_called_once_with('bedrock-runtime', 'us-west-2', creds)


class TestAmazonBedrockLLMClientGenerateResponse:
    @pytest.mark.asyncio
    async def test_text_response_uses_converse(self, bedrock_client, mock_boto3_client):
        mock_boto3_client.converse.return_value = converse_response('This is a test response')

        result = await bedrock_client._generate_response(messages=MESSAGES)

        assert result == {'content': 'This is a test response'}
        request = mock_boto3_client.converse.call_args.kwargs
        assert request['modelId'] == 'us.anthropic.claude-sonnet-4-20250514-v1:0'
        assert request['system'] == [{'text': 'System message'}]
        assert request['messages'] == [{'role': 'user', 'content': [{'text': 'User message'}]}]
        assert request['inferenceConfig'] == {'maxTokens': 16384, 'temperature': 0.5}
        mock_boto3_client.invoke_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_structured_response(self, bedrock_client, mock_boto3_client):
        mock_boto3_client.converse.return_value = converse_response(
            '{"test_field": "test_value", "optional_field": 42}'
        )

        result = await bedrock_client._generate_response(
            messages=MESSAGES, response_model=ResponseModel
        )

        assert result == {'test_field': 'test_value', 'optional_field': 42}

    @pytest.mark.asyncio
    async def test_reasoning_blocks_are_ignored(self, bedrock_client, mock_boto3_client):
        """gpt-oss returns reasoningContent next to the text block."""
        mock_boto3_client.converse.return_value = converse_response(
            '```json\n{"test_field": "v"}\n```', reasoning='thinking about {braces}'
        )

        result = await bedrock_client._generate_response(
            messages=MESSAGES, response_model=ResponseModel
        )

        assert result['test_field'] == 'v'

    @pytest.mark.asyncio
    async def test_model_override_and_small_model(self, mock_boto3_client):
        client = AmazonBedrockLLMClient(
            config=LLMConfig(model='big', small_model='small'), region='us-east-1'
        )
        client.client = mock_boto3_client
        mock_boto3_client.converse.return_value = converse_response('ok')

        await client._generate_response(MESSAGES, model_size=ModelSize.small)
        assert mock_boto3_client.converse.call_args.kwargs['modelId'] == 'small'

        await client._generate_response(MESSAGES, model='explicit')
        assert mock_boto3_client.converse.call_args.kwargs['modelId'] == 'explicit'

        await client._generate_response(MESSAGES)
        assert mock_boto3_client.converse.call_args.kwargs['modelId'] == 'big'

    @pytest.mark.asyncio
    async def test_public_generate_response_injects_schema_once(
        self, bedrock_client, mock_boto3_client
    ):
        mock_boto3_client.converse.return_value = converse_response(
            '{"test_field": "test_value", "optional_field": 1}'
        )

        result = await bedrock_client.generate_response(
            messages=[
                Message(role='system', content='System message'),
                Message(role='user', content='User message'),
            ],
            response_model=ResponseModel,
        )

        assert result['test_field'] == 'test_value'
        sent = mock_boto3_client.converse.call_args.kwargs['messages'][0]['content'][0]['text']
        assert sent.count('Respond with a JSON object in the following format:') == 1

    @pytest.mark.asyncio
    async def test_json_parsing_error(self, bedrock_client, mock_boto3_client):
        mock_boto3_client.converse.return_value = converse_response('Invalid JSON response')

        with pytest.raises((ValueError, json.JSONDecodeError)):
            await bedrock_client._generate_response(
                messages=[Message(role='user', content='Test message')],
                response_model=ResponseModel,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('code', ['ThrottlingException', 'ServiceQuotaExceededException'])
    async def test_throttling_raises_rate_limit_error(
        self, bedrock_client, mock_boto3_client, code
    ):
        mock_boto3_client.converse.side_effect = ClientError(
            {'Error': {'Code': code, 'Message': 'slow down'}}, 'Converse'
        )

        with pytest.raises(RateLimitError):
            await bedrock_client._generate_response(messages=MESSAGES)

    @pytest.mark.asyncio
    async def test_other_client_errors_propagate(self, bedrock_client, mock_boto3_client):
        mock_boto3_client.converse.side_effect = ClientError(
            {'Error': {'Code': 'AccessDeniedException', 'Message': 'no model access'}}, 'Converse'
        )

        with pytest.raises(ClientError):
            await bedrock_client._generate_response(messages=MESSAGES)

    @pytest.mark.asyncio
    async def test_clean_json_response(self, bedrock_client):
        assert bedrock_client._clean_json_response('```json\n{"test": "value"}\n```') == (
            '{"test": "value"}'
        )
        assert bedrock_client._clean_json_response('{{"test": "value"}}') == '{"test": "value"}'
        assert (
            bedrock_client._clean_json_response(
                'Here is the JSON: {"test": "value"} and some more text'
            )
            == '{"test": "value"}'
        )

    @pytest.mark.asyncio
    async def test_clean_json_keeps_valid_nested_objects(self, bedrock_client):
        nested = '{"attributes":{"name":"Alice"}}'
        assert bedrock_client._clean_json_response(nested) == nested
        assert bedrock_client._clean_json_response(f'```json\n{nested}\n```') == nested

    @pytest.mark.asyncio
    async def test_clean_json_keeps_braces_inside_strings(self, bedrock_client):
        text = '{"a":"x}y","b":{"c":"{"}}'
        assert json.loads(bedrock_client._clean_json_response(text + ' trailing')) == json.loads(
            text
        )

    @pytest.mark.asyncio
    async def test_clean_json_recovers_doubled_braces_only_when_needed(self, bedrock_client):
        assert bedrock_client._clean_json_response('{{"a": {"b": 1}}}') == '{"a": {"b": 1}}'


class TestAmazonBedrockLLMClientConfig:
    def test_config_max_tokens_is_kept_when_not_overridden(self):
        client = AmazonBedrockLLMClient(config=LLMConfig(model='m', max_tokens=1000))
        assert client.max_tokens == 1000

    def test_explicit_max_tokens_overrides_config(self):
        client = AmazonBedrockLLMClient(config=LLMConfig(model='m', max_tokens=1000), max_tokens=50)
        assert client.max_tokens == 50

    @pytest.mark.asyncio
    async def test_failures_do_not_log_model_output(
        self, bedrock_client, mock_boto3_client, caplog
    ):
        secret = 'SENSITIVE-EPISODE-TEXT'
        mock_boto3_client.converse.return_value = converse_response(f'not json {secret}')

        with caplog.at_level(logging.DEBUG), pytest.raises(ValueError):
            await bedrock_client._generate_response(messages=MESSAGES, response_model=ResponseModel)

        assert secret not in caplog.text

    @pytest.mark.asyncio
    async def test_invocation_errors_log_only_the_error_code(
        self, bedrock_client, mock_boto3_client, caplog
    ):
        mock_boto3_client.converse.side_effect = ClientError(
            {'Error': {'Code': 'ValidationException', 'Message': 'SENSITIVE-ECHO'}}, 'Converse'
        )

        with caplog.at_level(logging.DEBUG), pytest.raises(ClientError):
            await bedrock_client._generate_response(messages=MESSAGES)

        assert 'ValidationException' in caplog.text
        assert 'SENSITIVE-ECHO' not in caplog.text

    @pytest.mark.asyncio
    async def test_malformed_json_is_retried_by_the_base_policy(
        self, bedrock_client, mock_boto3_client
    ):
        from graphiti_core.llm_client.client import is_server_or_retry_error

        mock_boto3_client.converse.return_value = converse_response('{"test_field": ')

        with pytest.raises(json.JSONDecodeError) as exc:
            await bedrock_client._generate_response(messages=MESSAGES, response_model=ResponseModel)

        assert is_server_or_retry_error(exc.value)

    @pytest.mark.asyncio
    async def test_schema_mismatch_is_not_a_json_error(self, bedrock_client, mock_boto3_client):
        mock_boto3_client.converse.return_value = converse_response('{"unrelated": 1}')

        with pytest.raises(ValueError) as exc:
            await bedrock_client._generate_response(messages=MESSAGES, response_model=ResponseModel)

        assert not isinstance(exc.value, json.JSONDecodeError)

    @pytest.mark.asyncio
    async def test_clean_json_does_not_mangle_truncated_objects(self, bedrock_client):
        # A truncated object must stay an object-shaped string so it fails as bad JSON,
        # not decode as a bare string after dropping the first character.
        cleaned = bedrock_client._clean_json_response('{"test_field": ')
        with pytest.raises(json.JSONDecodeError):
            json.loads(cleaned)
