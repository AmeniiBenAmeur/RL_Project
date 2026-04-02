'''
Author: Maonan Wang
Date: 2025-04-24 10:02:21
Description: LLM Config for AI Agent (Ollama)
LastEditors: WANG Maonan
LastEditTime: 2025-07-30 13:39:00
'''
# Esempio di come deve essere il file llm_config.py
vlm_cfg = {
    'model_type': 'oai', # Fondamentale!
    'model': 'gpt-4o',   # O il modello VLM che stai usando
    'api_key': 'LA_TUA_CHIAVE_QUI',
}

llm_cfg = {
    'model_type': 'oai', # Fondamentale!
    'model': 'gpt-4o',
    'api_key': 'LA_TUA_CHIAVE_QUI',
}

llm_cfg_json = {
    'model_type': 'oai',
    'model': 'gpt-4o',
    'api_key': 'LA_TUA_CHIAVE_QUI',
    'generate_cfg': {'response_format': {'type': 'json_object'}}
}