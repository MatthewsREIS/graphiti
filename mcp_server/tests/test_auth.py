#!/usr/bin/env python3
"""Bearer-token auth for the MCP endpoint: token verification and HTTP behaviour."""

import base64
import hashlib
import hmac
import json
import sys
import time
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from mcp.server.mcpserver import MCPServer
from pydantic import ValidationError
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from services.auth import AuthConfig, JWTTokenVerifier, build_auth_kwargs

ISSUER = 'https://login.example.com/realms/graphiti-mcp'
AUDIENCE = 'https://graphiti.example.com/mcp'


def _key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


SIGNING_KEY = _key()
OTHER_KEY = _key()


def _token(key=SIGNING_KEY, algorithm='RS256', **overrides) -> str:
    now = int(time.time())
    claims = {
        'iss': ISSUER,
        'aud': AUDIENCE,
        'sub': 'user-1',
        'azp': 'claude-code',
        'scope': 'openid email',
        'iat': now,
        'exp': now + 300,
    }
    claims.update(overrides)
    claims = {name: value for name, value in claims.items() if value is not None}
    return jwt.encode(claims, key, algorithm=algorithm)


def _verifier() -> JWTTokenVerifier:
    public_key = SIGNING_KEY.public_key()
    return JWTTokenVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_url='https://login.example.com/unused',
        key_resolver=lambda token: public_key,
    )


@pytest.mark.asyncio
async def test_valid_token_is_accepted():
    access = await _verifier().verify_token(_token())
    assert access is not None
    assert access.subject == 'user-1'
    assert access.client_id == 'claude-code'
    assert access.scopes == ['openid', 'email']
    assert access.resource == AUDIENCE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'token',
    [
        pytest.param(_token(exp=int(time.time()) - 600), id='expired'),
        pytest.param(_token(aud='https://other.example.com/mcp'), id='wrong-audience'),
        pytest.param(_token(iss='https://evil.example.com/realms/x'), id='wrong-issuer'),
        pytest.param(_token(key=OTHER_KEY), id='signed-by-another-key'),
        pytest.param(_token(exp=None), id='no-expiry'),
        pytest.param(_token(sub=None), id='no-subject'),
        pytest.param('not-a-jwt', id='garbage'),
        pytest.param('', id='empty'),
    ],
)
async def test_bad_tokens_are_rejected(token: str):
    assert await _verifier().verify_token(token) is None


@pytest.mark.asyncio
async def test_unsigned_token_is_rejected():
    unsigned = jwt.encode(
        {'iss': ISSUER, 'aud': AUDIENCE, 'sub': 'u', 'exp': int(time.time()) + 60},
        None,
        algorithm='none',
    )
    assert await _verifier().verify_token(unsigned) is None


def _forged_hs256(secret: bytes, claims: dict) -> str:
    # PyJWT refuses to sign HS256 with a PEM key, so build the token by hand.
    def encode(part: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(part).encode()).rstrip(b'=').decode()

    signing_input = f'{encode({"alg": "HS256", "typ": "JWT"})}.{encode(claims)}'
    signature = hmac.new(secret, signing_input.encode(), hashlib.sha256).digest()
    return f'{signing_input}.{base64.urlsafe_b64encode(signature).rstrip(b"=").decode()}'


@pytest.mark.asyncio
async def test_hmac_token_signed_with_the_public_key_is_rejected():
    # Algorithm confusion: an HS256 token keyed with the published RSA public key.
    public_pem = SIGNING_KEY.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    forged = _forged_hs256(
        public_pem,
        {'iss': ISSUER, 'aud': AUDIENCE, 'sub': 'attacker', 'exp': int(time.time()) + 300},
    )
    assert await _verifier().verify_token(forged) is None


@pytest.mark.asyncio
async def test_unreachable_key_set_rejects_instead_of_raising():
    def unreachable(token: str):
        raise jwt.PyJWKClientConnectionError('jwks unreachable')

    verifier = JWTTokenVerifier(
        issuer=ISSUER, audience=AUDIENCE, jwks_url='https://x', key_resolver=unreachable
    )
    assert await verifier.verify_token(_token()) is None


def test_auth_is_off_when_nothing_is_configured():
    assert build_auth_kwargs(AuthConfig(_env_file=None)) == {}


@pytest.mark.parametrize('present', ['issuer_url', 'resource_url'])
def test_half_configured_auth_fails_instead_of_running_open(present: str):
    values = {'issuer_url': ISSUER, 'resource_url': AUDIENCE}
    with pytest.raises(ValidationError):
        AuthConfig(_env_file=None, **{present: values[present]})


def test_jwks_url_defaults_to_the_keycloak_certs_endpoint():
    config = AuthConfig(_env_file=None, issuer_url=ISSUER + '/', resource_url=AUDIENCE)
    assert config.jwks == ISSUER + '/protocol/openid-connect/certs'


def _app() -> TestClient:
    public_key = SIGNING_KEY.public_key()
    kwargs = build_auth_kwargs(
        AuthConfig(_env_file=None, issuer_url=ISSUER, resource_url=AUDIENCE),
        key_resolver=lambda token: public_key,
    )
    server = MCPServer('test', **kwargs)

    @server.custom_route('/health', methods=['GET'])
    async def health(request):
        return JSONResponse({'status': 'healthy'})

    return TestClient(server.streamable_http_app(), base_url='http://localhost:8000')


_INITIALIZE = {
    'jsonrpc': '2.0',
    'id': 1,
    'method': 'initialize',
    'params': {
        'protocolVersion': '2025-03-26',
        'capabilities': {},
        'clientInfo': {'name': 'test', 'version': '1'},
    },
}
_HEADERS = {'Accept': 'application/json, text/event-stream', 'Content-Type': 'application/json'}


def test_health_stays_open_but_the_mcp_endpoint_requires_a_token():
    with _app() as client:
        assert client.get('/health').status_code == 200

        denied = client.post('/mcp', json=_INITIALIZE, headers=_HEADERS)
        assert denied.status_code == 401
        assert 'resource_metadata' in denied.headers['www-authenticate']

        bad = client.post(
            '/mcp', json=_INITIALIZE, headers={**_HEADERS, 'Authorization': 'Bearer nope'}
        )
        assert bad.status_code == 401

        allowed = client.post(
            '/mcp', json=_INITIALIZE, headers={**_HEADERS, 'Authorization': f'Bearer {_token()}'}
        )
        assert allowed.status_code == 200


def test_discovery_document_points_clients_at_the_issuer():
    with _app() as client:
        response = client.get('/.well-known/oauth-protected-resource/mcp')
        assert response.status_code == 200
        document = response.json()
        assert document['resource'] == AUDIENCE
        assert [server.rstrip('/') for server in document['authorization_servers']] == [ISSUER]
