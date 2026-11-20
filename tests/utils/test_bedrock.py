import os
from unittest.mock import patch

import pytest

from graphiti_core.utils.bedrock import (
    BEDROCK_BEARER_TOKEN_ENV,
    BedrockCredentials,
    create_bedrock_client,
)


@pytest.fixture
def boto3_client():
    with patch('boto3.client') as m:
        yield m


def test_default_chain_when_no_credentials(boto3_client):
    create_bedrock_client('bedrock-runtime', 'us-east-1')

    kwargs = boto3_client.call_args.kwargs
    assert boto3_client.call_args.args == ('bedrock-runtime',)
    assert kwargs['region_name'] == 'us-east-1'
    assert 'aws_access_key_id' not in kwargs


def test_explicit_keys_and_session_token(boto3_client):
    creds = BedrockCredentials(access_key_id='AKIA', secret_access_key='s', session_token='t')

    create_bedrock_client('bedrock-runtime', 'us-west-2', creds)

    kwargs = boto3_client.call_args.kwargs
    assert kwargs['aws_access_key_id'] == 'AKIA'
    assert kwargs['aws_secret_access_key'] == 's'
    assert kwargs['aws_session_token'] == 't'


def test_access_key_without_secret_is_rejected(boto3_client):
    with pytest.raises(ValueError):
        create_bedrock_client('bedrock-runtime', 'us-east-1', BedrockCredentials(access_key_id='A'))


def test_profile_uses_session():
    with patch('boto3.Session') as session:
        create_bedrock_client('bedrock-runtime', 'us-east-1', BedrockCredentials(profile='dev'))

    session.assert_called_once_with(profile_name='dev')
    session.return_value.client.assert_called_once()


def test_bearer_token_sets_env_and_wins_over_keys(boto3_client):
    creds = BedrockCredentials(bearer_token='tok', access_key_id='A', secret_access_key='s')
    with patch.dict(os.environ, {}, clear=False):
        create_bedrock_client('bedrock-runtime', 'us-east-1', creds)
        assert os.environ[BEDROCK_BEARER_TOKEN_ENV] == 'tok'
        os.environ.pop(BEDROCK_BEARER_TOKEN_ENV)

    assert 'aws_access_key_id' not in boto3_client.call_args.kwargs


def test_bearer_token_wins_over_partial_iam_keys(boto3_client):
    creds = BedrockCredentials(bearer_token='tok', access_key_id='A')
    with patch.dict(os.environ, {}, clear=False):
        create_bedrock_client('bedrock-runtime', 'us-east-1', creds)
        os.environ.pop(BEDROCK_BEARER_TOKEN_ENV, None)

    assert 'aws_access_key_id' not in boto3_client.call_args.kwargs


def test_agent_runtime_client_uses_iam_keys_even_with_a_bearer_token(boto3_client):
    # bedrock-agent-runtime (Rerank) does not accept Bedrock API keys
    creds = BedrockCredentials(bearer_token='tok', access_key_id='A', secret_access_key='s')
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop(BEDROCK_BEARER_TOKEN_ENV, None)
        create_bedrock_client('bedrock-agent-runtime', 'us-east-1', creds)
        assert BEDROCK_BEARER_TOKEN_ENV not in os.environ

    kwargs = boto3_client.call_args.kwargs
    assert kwargs['aws_access_key_id'] == 'A'
    assert kwargs['aws_secret_access_key'] == 's'


def test_agent_runtime_client_uses_profile_even_with_a_bearer_token():
    creds = BedrockCredentials(bearer_token='tok', profile='dev')
    with patch('boto3.Session') as session:
        create_bedrock_client('bedrock-agent-runtime', 'us-east-1', creds)

    session.assert_called_once_with(profile_name='dev')


def test_runtime_client_still_prefers_the_bearer_token(boto3_client):
    creds = BedrockCredentials(bearer_token='tok', access_key_id='A', secret_access_key='s')
    with patch.dict(os.environ, {}, clear=False):
        create_bedrock_client('bedrock-runtime', 'us-east-1', creds)
        assert os.environ[BEDROCK_BEARER_TOKEN_ENV] == 'tok'
        os.environ.pop(BEDROCK_BEARER_TOKEN_ENV)

    assert 'aws_access_key_id' not in boto3_client.call_args.kwargs
