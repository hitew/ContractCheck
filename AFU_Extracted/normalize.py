def normalize_doc_tokens(doc_tokens, doc_tokens_type):
    return [normalize_tokens(tokens, token_type) for tokens, token_type in zip(doc_tokens, doc_tokens_type)]

def normalize_tokens(token_list, token_type):
    token_list_normalized = []
    for i, token in enumerate(token_list):
        if token_type[i] in ['117', '118', '2', '15', '34']:
            continue
        elif token_type[i] == '95':
            token = "VersionLiteral"
        elif token_type[i] in ['115', '97', '98', '100']:
            token = "normalizedToken"
        token_list_normalized.append(token.lower() if not token.startswith('%%%%') else token)
    return token_list_normalized
