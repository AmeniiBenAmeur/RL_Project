import copy
import json
from qwen_agent import Agent
from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message, ContentItem
from typing import Dict, Iterator, List

# --- CONFIGURAZIONI LOCALI (Fix 127.0.0.1 per Windows) ---
# Usiamo 127.0.0.1 invece di localhost per evitare rifiuti di connessione
llm_cfg = {
    'model': 'Qwen/Qwen2.5-72B-Instruct-AWQ',
    'model_type': 'oai',
    'model_server': 'http://127.0.0.1:5070/v1',
    'api_key': 'token-abc123',
}

vlm_cfg = {
    'model': 'Qwen/Qwen2.5-VL-32B-Instruct-AWQ',
    'model_type': 'qwenvl_oai',
    'model_server': 'http://127.0.0.1:5030/v1',
    'api_key': 'token-abc123',
}

llm_cfg_json = {
    'model': 'Qwen/Qwen2.5-72B-Instruct-AWQ',
    'model_type': 'oai',
    'model_server': 'http://127.0.0.1:5070/v1',
    'api_key': 'token-abc123',
    'generate_cfg': {'response_format': {"type": "json_object"}}
}

# --- DEFINIZIONE AGENTI ---

# Agente per la comprensione visiva della scena
scene_understanding_agent = Assistant(
    name='Scene Understanding Agent',
    llm=vlm_cfg,
    system_message=(
        "Act as a traffic monitoring system. Describe road conditions, "
        "including congestion levels and identify any confirmed special vehicles "
        "(ambulance, police, fire truck) with clear identifiers."
    )
)

# Agente per l'analisi globale del traffico
scene_analysis_agent = Assistant(
    llm=llm_cfg,
    system_message=(
        "You are a traffic police officer. Summarize the status of all traffic phases "
        "and determine if emergency conditions are present."
    )
)

# Agente per la selezione della modalità (Emergenza vs Normale)
mode_select_agent = Assistant(
    llm=llm_cfg,
    system_message=(
        "Select exactly one agent based on the junction's status: "
        "[Emergency Case Decision Agent, Normal Case Decision Agent]. "
        "Return ONLY the agent name."
    )
)

# Classe per la gestione dei casi critici
class ConcernCaseAgent(Agent):
    def __init__(self, phase_num: int, llm_cfg: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_traffic_phase = [f"Phase-{i}" for i in range(phase_num)]
        
        # Agente decisionale interno
        self.decision_agent = Assistant(
            llm=llm_cfg, 
            system_message="Prioritize emergency vehicles and optimize flow."
        )
        
        # Agente di controllo output JSON
        self.check_agent = Assistant(
            llm=llm_cfg_json, 
            system_message=(
                f"Valid phases: {self.available_traffic_phase}. "
                "Output MUST be valid JSON with 'decision' and 'explanation' keys."
            )
        )

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        new_messages = copy.deepcopy(messages)
        res = []
        
        # Step 1: Generazione Decisione
        for rsp in self.decision_agent.run(new_messages):
            yield res + rsp
        
        # Aggiornamento contesto
        if rsp:
            res.extend(rsp)
            new_messages.extend(rsp)
        
        # Step 2: Validazione e Formattazione JSON
        for rsp in self.check_agent.run(new_messages):
            yield res + rsp