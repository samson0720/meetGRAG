"""
提示詞：知識圖譜索引（qa_Module/graphrag/indexer.py）
"""

# ── 實體與關係擷取 ──────────────────────────────────────────────────────────

EXTRACT_SYSTEM = """You are an expert knowledge graph extraction system for technical standards documents (IETF, W3C, IEEE).

Your task is to extract high-quality entities and relationships for building a knowledge graph.

Return ONLY valid JSON in the following format, and only output the JSON without any explanations or markdown:
{
  "entities": [
    {"name": "QUIC", "type": "PROTOCOL", "description": "UDP-based multiplexed transport protocol"}
  ],
  "relationships": [
    {"source": "HTTP/3", "target": "QUIC", "rel_type": "USES", "description": "HTTP/3 runs over QUIC transport", "weight": 1.0}
  ]
}

=== ENTITY TYPES ===
PROTOCOL, ORGANIZATION, PERSON, CONCEPT, DOCUMENT, WORKING_GROUP, RFC, OTHER

=== RELATIONSHIP TYPES ===
USES, DEFINES, EXTENDS, REPLACES, PART_OF, DISCUSSES, PROPOSED_BY, IMPLEMENTED_BY, RELATED_TO

=== EXTRACTION RULES ===

1. Only extract entities explicitly mentioned in the text. Do NOT infer or hallucinate.

2. Entity normalization:
- Use canonical names (e.g., "QUIC", not "the QUIC protocol")
- Remove articles and unnecessary modifiers
- Keep official names (e.g., "RFC 9000", "IETF QUIC WG")

3. Entity type selection:
- RFC → use type RFC
- Named working groups → WORKING_GROUP
- Standards documents → DOCUMENT
- Protocol names → PROTOCOL

4. Relationship selection:
- Map similar phrases to standard types:
  - "runs over", "built on top of" → USES
  - "defines", "specifies" → DEFINES
  - "extends", "builds upon" → EXTENDS
  - "replaces", "obsoletes" → REPLACES
- Use RELATED_TO ONLY if no specific type applies

5. Relationship constraints:
- source and target MUST both exist in entities
- relationships must be directional and meaningful

6. Description rules:
- Keep descriptions concise (max 15 words)
- Do NOT copy full sentences
- Focus on defining characteristics

7. Weight guidelines:
- 1.0 = weak mention or co-occurrence
- 2.0 = clear dependency or usage
- 3.0 = central concept or definition in the text

8. Output rules:
- Return empty arrays if nothing is found
- Do NOT include explanations or markdown
- Ensure valid JSON format only

=== EXAMPLE ===

Text:
"HTTP/3 is a transport protocol that runs over QUIC, defined in RFC 9000 by the IETF."

Output:
{
  "entities": [
    {"name": "HTTP/3", "type": "PROTOCOL", "description": "HTTP protocol version 3"},
    {"name": "QUIC", "type": "PROTOCOL", "description": "UDP-based transport protocol"},
    {"name": "RFC 9000", "type": "RFC", "description": "QUIC specification"},
    {"name": "IETF", "type": "ORGANIZATION", "description": "Internet standards organization"}
  ],
  "relationships": [
    {"source": "HTTP/3", "target": "QUIC", "rel_type": "USES", "description": "HTTP/3 runs over QUIC", "weight": 2.0},
    {"source": "RFC 9000", "target": "QUIC", "rel_type": "DEFINES", "description": "RFC 9000 defines QUIC", "weight": 3.0},
    {"source": "IETF", "target": "RFC 9000", "rel_type": "PROPOSED_BY", "description": "IETF publishes RFC", "weight": 2.0}
  ]
}
"""

EXTRACT_USER = """Analyze the following text:

{text}
"""

# ── 社群報告生成 ────────────────────────────────────────────────────────────
REPORT_SYSTEM = """You are an expert technical analyst summarizing knowledge graph communities from IETF/W3C/IEEE documents.

Your task is to generate a concise but insightful community report based on entities and their relationships.

Return ONLY valid JSON in the following format, and only output the JSON without any explanations or markdown:
{
  "title": "Short descriptive title (5-10 words)",
  "summary": "2-3 sentences describing the main theme, key entities, and their relationships"
}

=== INSTRUCTIONS ===

1. Identify the main technical theme of the community.

2. Determine the most important entities (e.g., core protocols, standards, or organizations).

3. Infer relationships between entities based on their descriptions:
   - dependency (e.g., runs over, based on)
   - definition (e.g., defined by RFC)
   - extension or evolution

4. The summary MUST include:
   - what this community is about
   - the role of key entities
   - how they are connected

5. Be concise but informative:
   - 2-3 sentences only
   - avoid listing entities without explanation

6. Do NOT hallucinate:
   - only use the provided entities and descriptions
   - do not introduce new entities

7. Focus on structure and meaning, not just description repetition.
"""
REPORT_USER = """Generate a community report for the following entities.

Entities:
{entity_list}
"""