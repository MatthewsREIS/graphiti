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

import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    import boto3
    from botocore.config import Config
else:
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise ImportError(
            'boto3 is required for Amazon Bedrock support. '
            'Install it with: pip install graphiti-core[bedrock]'
        ) from None

BEDROCK_BEARER_TOKEN_ENV = 'AWS_BEARER_TOKEN_BEDROCK'
# Bedrock API keys work only with these services. Others (e.g. bedrock-agent-runtime, used for
# Rerank) need IAM credentials, so they ignore the bearer token.
BEARER_TOKEN_SERVICES = ('bedrock', 'bedrock-runtime')


def error_label(e: BaseException) -> str:
    """AWS error code, or the exception type. Safe to log: it never includes the message,
    which can echo request content such as episode text."""
    response = getattr(e, 'response', None)
    code = response.get('Error', {}).get('Code') if isinstance(response, dict) else None
    return code or type(e).__name__


class BedrockCredentials(BaseModel):
    """Explicit AWS credentials for Bedrock. Every field is optional.

    Precedence: bearer_token, then access_key_id/secret_access_key(/session_token),
    then profile. When nothing is set, boto3's default chain applies (env vars,
    shared config, SSO, IRSA, instance role).
    """

    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    profile: str | None = None
    # Bedrock API key (long- or short-term). boto3 reads it from AWS_BEARER_TOKEN_BEDROCK.
    bearer_token: str | None = None


def create_bedrock_client(
    service: str,
    region: str,
    credentials: BedrockCredentials | None = None,
    **client_kwargs: Any,
):
    """Create a boto3 client (bedrock-runtime, bedrock-agent-runtime) for the given credentials."""
    creds = credentials or BedrockCredentials()

    client_kwargs.setdefault('config', Config(read_timeout=300, retries={'max_attempts': 3}))

    if creds.bearer_token and service in BEARER_TOKEN_SERVICES:
        # ponytail: botocore only reads the Bedrock API key from the environment, so this is
        # process-wide. Fine for one Bedrock identity per process.
        os.environ[BEDROCK_BEARER_TOKEN_ENV] = creds.bearer_token
        return boto3.client(service, region_name=region, **client_kwargs)

    if bool(creds.access_key_id) != bool(creds.secret_access_key):
        raise ValueError('Bedrock access_key_id and secret_access_key must be set together')

    if creds.access_key_id:
        return boto3.client(
            service,
            region_name=region,
            aws_access_key_id=creds.access_key_id,
            aws_secret_access_key=creds.secret_access_key,
            aws_session_token=creds.session_token,
            **client_kwargs,
        )

    if creds.profile:
        return boto3.Session(profile_name=creds.profile).client(
            service, region_name=region, **client_kwargs
        )

    return boto3.client(service, region_name=region, **client_kwargs)
