"""
Base codes for Agent and its internal state
"""

from enum import Enum
import uuid


class Status(Enum):
    """
    Agent status, following the SIR model
    """
    Susceptible = 'S'  # Suscetíveis
    Exposed = 'E'  # Expostos
    PreSymptomatic = 'P'  # Pré-sintomáticos
    Asymptomatic = 'A'  # Assintomáticos
    Infected_Mild = 'I1'  # Infectados com sintomas leves
    Infected_Severe = 'I2'  # Infectados com sintomas severos
    Infected_Critical = 'I3'  # Infectados críticos
    Recovered_Mild = 'R1'  # Recuperados de sintomas leves
    Recovered_Severe = 'R2'  # Recuperados de sintomas severos
    Recovered_Critical = 'R3'  # Recuperados de sintomas críticos
    Death = 'O'  # Mortos


class AgentType(Enum):
    """
    The type of the agent, or the node at the Graph
    """
    Individual = 'Individual'
    House = 'Household'
    Business = 'Business_Service'
    Government = 'Government'
    Healthcare = 'Healthcare_Facility'


class Agent:
    def __init__(self, agent_type=AgentType.Individual, **kwargs):
        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.x = kwargs.get('x', 0)
        self.y = kwargs.get('y', 0)
        self.status = kwargs.get('status', Status.Susceptible)
        self.infected_status = kwargs.get('status', Status.Infected_Mild)
        self.agent_type = agent_type
        self.infected_time = kwargs.get('infected_time', 0)
        self.age = kwargs.get('age', 0)
        self.social_stratum = kwargs.get('social_stratum', 0)
        self.wealth = kwargs.get('wealth', 0.0)
        self.environment = kwargs.get('environment', None)
        # Taxa de subnotificação
        self.subnotificacao = kwargs.get('subnotificacao', 0.0)
        # Probabilidades de progressão da doença e mortalidade, adaptadas para o modelo SEPAI3R3O
        self.prob_progressao = kwargs.get('prob_progressao', {
            'E': 0.1, 'P': 0.2, 'A': 0.05, 'I1': 0.1, 'I2': 0.2, 'I3': 0.3
        })
        self.prob_morte = 0.5

    def __str__(self):
        return f"Agent(ID: {self.id}, Type: {self.agent_type.name}, Status: {self.status.name}, Position: ({self.x}, {self.y}))"