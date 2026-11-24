#!/usr/bin/env python3
"""Entity types must accept a null attribute instead of failing the whole episode."""

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import entity_types
from models.entity_types import ENTITY_TYPES


def _entity_models() -> list[type[BaseModel]]:
    return [
        value
        for value in vars(entity_types).values()
        if isinstance(value, type)
        and issubclass(value, BaseModel)
        and value is not BaseModel
        and value.__module__ == entity_types.__name__
    ]


@pytest.mark.parametrize('model', _entity_models(), ids=lambda model: model.__name__)
def test_every_attribute_may_be_null(model: type[BaseModel]):
    # gpt-oss returns null for an attribute the context does not support (a Requirement
    # with no named project); a required string failed the whole episode.
    model.model_validate({name: None for name in model.model_fields})


def test_entity_type_registry_covers_every_model():
    assert {model.__name__ for model in _entity_models()} <= set(ENTITY_TYPES)
