from __future__ import annotations

from modules.schema_agent import SchemaDetection


def is_relevant_codegen_dataset(schema: SchemaDetection) -> bool:
    return bool(schema.is_codegen_dataset)

