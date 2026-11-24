"""Bearer-token auth for the MCP endpoint.

The server is an OAuth resource server: clients obtain a token from an OIDC issuer (Keycloak) and
this module only checks that a token was signed by that issuer and issued for this server.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import jwt
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

KeyResolver = Callable[[str], Any]


class AuthConfig(BaseSettings):
    """Read from MCP_AUTH_* at import time, because the MCP server object is built then.

    Both URLs or neither: a half-set config must stop startup, never leave the endpoint open.
    """

    model_config = SettingsConfigDict(env_prefix='MCP_AUTH_', extra='ignore')

    issuer_url: str | None = None
    resource_url: str | None = None
    jwks_url: str | None = None

    @model_validator(mode='after')
    def _both_or_neither(self) -> 'AuthConfig':
        if bool(self.issuer_url) != bool(self.resource_url):
            raise ValueError('MCP_AUTH_ISSUER_URL and MCP_AUTH_RESOURCE_URL must be set together')
        return self

    @property
    def enabled(self) -> bool:
        return bool(self.issuer_url and self.resource_url)

    @property
    def jwks(self) -> str:
        return (
            self.jwks_url or f'{(self.issuer_url or "").rstrip("/")}/protocol/openid-connect/certs'
        )


class JWTTokenVerifier:
    """Accepts access tokens signed by one issuer for one audience."""

    def __init__(
        self,
        issuer: str,
        audience: str,
        jwks_url: str,
        algorithms: tuple[str, ...] = ('RS256',),
        key_resolver: KeyResolver | None = None,
    ):
        self._issuer = issuer
        self._audience = audience
        self._algorithms = list(algorithms)
        if key_resolver is None:
            jwks_client = jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=300, timeout=10)
            key_resolver = lambda token: jwks_client.get_signing_key_from_jwt(token).key  # noqa: E731
        self._resolve_key = key_resolver

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            # PyJWKClient fetches the key set over blocking HTTP, so keep it off the event loop.
            key = await asyncio.to_thread(self._resolve_key, token)
            claims = jwt.decode(
                token,
                key,
                algorithms=self._algorithms,
                audience=self._audience,
                issuer=self._issuer,
                options={'require': ['exp', 'iss', 'aud', 'sub']},
            )
        except jwt.PyJWTError as error:
            # Type only: never log the token or its claims.
            logger.info(f'Rejected bearer token ({type(error).__name__})')
            return None

        return AccessToken(
            token=token,
            client_id=claims.get('azp') or claims.get('client_id') or 'unknown',
            scopes=str(claims.get('scope', '')).split(),
            expires_at=int(claims['exp']),
            resource=self._audience,
            subject=str(claims['sub']),
        )


def build_auth_kwargs(
    config: AuthConfig | None = None, key_resolver: KeyResolver | None = None
) -> dict[str, Any]:
    """Keyword arguments for MCPServer: empty when auth is not configured."""
    config = config or AuthConfig()
    if not config.enabled:
        return {}
    assert config.issuer_url and config.resource_url
    return {
        'auth': AuthSettings(
            issuer_url=config.issuer_url,
            resource_server_url=config.resource_url,
            # The verifier checks the audience itself.
            validate_token_resource=False,
        ),
        'token_verifier': JWTTokenVerifier(
            issuer=config.issuer_url,
            audience=config.resource_url,
            jwks_url=config.jwks,
            key_resolver=key_resolver,
        ),
    }
