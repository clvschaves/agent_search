"""
Main code for Agent Based Simulation
"""

from covid_abs.agents import Status, Agent
from covid_abs.common import *
import numpy as np

def distance(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

class Simulation(object):
    def __init__(self, **kwargs):
        self.population = []  # A população de agentes
        self.population_size = kwargs.get("population_size", 20)  # Número de agentes
        self.length = kwargs.get("length", 10)  # Comprimento do ambiente compartilhado
        self.height = kwargs.get("height", 10)  # Altura do ambiente compartilhado
        self.initial_infected_perc = kwargs.get("initial_infected_perc", 0.05)  # Percentual inicial de infectados
        self.initial_immune_perc = kwargs.get("initial_immune_perc", 0.05)  # Percentual inicial de imunes
        self.initial_pre_symptomatic = kwargs.get("initial_pre_symptomatic", 0.05) # Percentual inicial de preSyntomatic
        self.initial_asymptomatic =  kwargs.get("initial_asymptomatic", 0.05) # Percentual inicial assintomático
        self.contagion_distance = kwargs.get("contagion_distance", 1.0)  # Distância mínima para contágio
        self.contagion_rate = kwargs.get("contagion_rate", 0.9)  # Probabilidade de contágio
        self.critical_limit = kwargs.get("critical_limit", 0.6)  # Limite crítico para o sistema de saúde
        self.amplitudes = kwargs.get('amplitudes', {
            'S': 5, 'E': 5, 'P': 5, 'A': 5, 'I1': 5, 'I2': 5, 'I3': 5, 'R1': 5, 'R2': 5, 'R3': 5, 'O': 0
        })  # Mobilidade média por estado
        self.minimum_income = kwargs.get("minimum_income", 1.0)  # Renda diária mínima
        self.minimum_expense = kwargs.get("minimum_expense", 1.0)  # Despesa diária mínima
        self.statistics = None  # Estatísticas da população atual
        self.triggers_simulation = kwargs.get("triggers_simulation", [])  # Alterações condicionais nos atributos da simulação
        self.triggers_population = kwargs.get("triggers_population", [])  # Alterações condicionais nos atributos da população
        self.total_wealth = kwargs.get("total_wealth", 10 ** 4)  # Riqueza total
        self.subnotification_rate = kwargs.get("subnotification_rate", 0.9)  # Taxa de subnotificação

    # Métodos adicionais da classe aqui...

    def _xclip(self, x):
        return np.clip(int(x), 0, self.length)

    def _yclip(self, y):
        return np.clip(int(y), 0, self.height)

    def get_population(self):
        """
        Return the population in the current iteration

        :return: a list with the current agent instances
        """
        return self.population

    def set_population(self, pop):
        """
        Update the population in the current iteration
        """
        self.population = pop

    def set_amplitudes(self, amp):
        self.amplitudes = amp

    def append_trigger_simulation(self, condition, attribute, action):
        """
        Append a conditional change in the Simulation attributes

        :param condition: a lambda function that receives the current simulation instance and
        returns a boolean
        :param attribute: string, the attribute name of the Simulation which will be changed
        :param action: a lambda function that receives the current simulation instance and returns
        the new value of the attribute
        """
        self.triggers_simulation.append({'condition': condition, 'attribute': attribute, 'action': action})

    def append_trigger_population(self, condition, attribute, action):
        """
        Append a conditional change in the population attributes

        :param condition: a lambda function that receives the current agent instance and returns a boolean
        :param attribute: string, the attribute name of the agent which will be changed
        :param action: a lambda function that receives the current agent instance and returns the new
        value of the attribute
        """
        self.triggers_population.append({'condition': condition, 'attribute': attribute, 'action': action})

    def random_position(self):
        x = self._xclip(self.length / 2 + (np.random.randn(1) * (self.length / 3)))
        y = self._yclip(self.height / 2 + (np.random.randn(1) * (self.height / 3)))

        return x, y

    def create_agent(self, status):
        """
        Create a new agent with the given status and incorporate the underreporting rate
        :param status: a value of agents.Status enum
        :return: the newly created agent
        """
        x, y = self.random_position()

        # Additional attributes for underreporting
        prob_subnotificacao = 0.1  # Example underreporting rate, adjust based on your model's needs

        age = int(np.random.beta(2, 5, 1) * 100)
        social_stratum = int(np.random.rand(1) * 100 // 20)
        
        # Create an agent with adjusted probabilities to reflect underreporting
        agent = Agent(
            x=x, y=y, age=age, status=status, social_stratum=social_stratum,
            prob_subnotificacao=prob_subnotificacao  # Pass the underreporting rate to the agent
        )
        self.population.append(agent)
        #print(f'populacao = {self.population[0].social_stratum}')

    def initialize(self):
        """
        Inicializa a simulação criando a população de agentes.
        Leva em consideração a taxa de subnotificação e a realidade socioeconômica da Ilha de Joana Bezerra, no Recife.
        """

        # População inicial infectada leve
        num_infectados_iniciais = int(self.population_size * self.initial_infected_perc)
        
        # População inicial infectada medio
        num_infectados_iniciais_medio = int(self.population_size * self.initial_infected_perc)

        # População inicial infectada grave
        num_infectados_iniciais_grave = int(self.population_size * self.initial_infected_perc)

        # População PreAssintomática inicial 
        num_initial_pre_symptomatic = int(self.initial_pre_symptomatic * self.population_size) 

        # População Assintomática inicial 
        num_initial_asymptomatic = int(self.initial_asymptomatic * self.population_size)
        
        for _ in range(num_infectados_iniciais + num_infectados_iniciais_medio + num_infectados_iniciais_grave):
            self.create_agent(Status.Exposed)  # Considera a taxa de subnotificação inicializando com expostos

        # População inicial imune
        num_imunes_iniciais = int(self.population_size * self.initial_immune_perc)
                
        for _ in range(num_imunes_iniciais):
            self.create_agent(Status.Recovered_Mild)  # Assumindo que a imunidade inicial vem de infecções leves

        for _ in range(num_imunes_iniciais):
            self.create_agent(Status.Recovered_Severe)

        for _ in range(num_imunes_iniciais):
            self.create_agent(Status.Infected_Critical)

        # População inicial suscetível
        for _ in range(self.population_size - len(self.population)):
            self.create_agent(Status.Susceptible)

        for _ in range(num_initial_pre_symptomatic):
            agent = self.create_agent(Status.PreSymptomatic)
           
        # Inicializando agentes 'Asymptomatic'
        for _ in range(num_initial_asymptomatic):
            agent = self.create_agent(Status.Asymptomatic)
            

        # Distribuição da riqueza com base nas condições socioeconômicas locais
        # Adaptação para refletir a realidade econômica da Ilha de Joana Bezerra
        total_wealth_local = self.total_wealth * 0.8  # Ajuste para refletir uma menor riqueza econômica
        for quintile in [0, 1, 2,3,4]:
            total = lorenz_curve[quintile] * total_wealth_local
            qty = max(1.0, sum(1 for a in self.population if a.social_stratum == quintile and a.age >= 18))
            ag_share = total / qty
            for agent in filter(lambda x: x.social_stratum == quintile and x.age >= 18, self.population):
                agent.wealth = ag_share
        

    def contact(self, agent1, agent2):
        """
        Realiza ações necessárias quando dois agentes entram em contato.

        :param agent1: uma instância de Agent
        :param agent2: uma instância de Agent
        """
        if agent1.status == Status.Susceptible:
            if agent2.status in [Status.Exposed, Status.PreSymptomatic, Status.Asymptomatic, Status.Infected_Mild, Status.Infected_Severe, Status.Infected_Critical]:
                contagion_chance = np.random.random()
                # Considera a taxa de subnotificação e ajusta a probabilidade de contágio
                adjusted_contagion_rate = self.contagion_rate * (1 - self.subnotification_rate)
                if contagion_chance <= adjusted_contagion_rate:
                    agent1.status = Status.Exposed  # O agente suscetível agora está exposto

        # Repete a lógica para o agente2 se o agente1 for o infectado
        if agent2.status == Status.Susceptible:
            if agent1.status in [Status.Exposed, Status.PreSymptomatic, Status.Asymptomatic, Status.Infected_Mild, Status.Infected_Severe, Status.Infected_Critical]:
                contagion_chance = np.random.random()
                # Considera a taxa de subnotificação e ajusta a probabilidade de contágio
                adjusted_contagion_rate = self.contagion_rate * (1 - self.subnotification_rate)
                if contagion_chance <= adjusted_contagion_rate:
                    agent2.status = Status.Exposed  # O agente suscetível agora está exposto


    def move(self, agent, triggers=[]):
        """
        Realiza as ações relacionadas ao movimento dos agentes no ambiente compartilhado.
        Este método é adaptado para refletir a heterogeneidade dos agentes e suas interações
        dinâmicas, levando em conta as características individuais e o contexto social e espacial.

        :param agent: uma instância de Agent.
        :param triggers: a lista de gatilhos populacionais relacionados ao movimento.
        """

        # Impede o movimento de agentes no estado de Morte ou em estados críticos de Infecção
        if agent.status in [Status.Death, Status.Infected_Critical]:
            return

        # Aplica gatilhos condicionais para movimento, se houver
        for trigger in triggers:
            if trigger['condition'](agent):
                agent.x, agent.y = trigger['action'](agent)
                return

        # Movimento baseado em características individuais e fatores estocásticos
        print(self.amplitudes[agent.status.value])
        ix = np.random.normal(loc=0, scale=self.amplitudes[agent.status.value])  # Desvio baseado no estado do agente
        iy = np.random.normal(loc=0, scale=self.amplitudes[agent.status.value])

        # Aplica os limites do ambiente para evitar que agentes saiam da área definida
        agent.x = max(0, min(self.length, agent.x + ix))
        agent.y = max(0, min(self.height, agent.y + iy))

        # Considera o impacto social e econômico do movimento
        interaction_factor = np.random.uniform(0.5, 1.5)  # Fator estocástico para interações sociais
        economic_impact = interaction_factor * self.minimum_expense * agent.wealth
        dist = np.sqrt(ix ** 2 + iy ** 2)
        result_ecom = np.random.rand(1)
        #agent.wealth = max(0, agent.wealth - economic_impact)  # Assegura que a riqueza não seja negativa
        agent.wealth += dist * interaction_factor * self.minimum_expense * basic_income[agent.social_stratum] 
       
       

    def update(self, agent):
        """
        Atualiza o estado do agente com base no modelo SEPAI3R3O, subnotificação e idade.
        
        :param agent: uma instância de Agent.
        """

        if agent.status == Status.Death:
            return  # Não atualiza se o agente já estiver morto

        # Atualiza o tempo de infecção se o agente estiver em um estado infeccioso
        if agent.status in [Status.Exposed, Status.PreSymptomatic, Status.Asymptomatic,
                            Status.Infected_Mild, Status.Infected_Severe, Status.Infected_Critical]:
            agent.infected_time += 1

        # Aplicando a subnotificação para ajustar a probabilidade de transição
        # Supondo que 'subnotificacao_rate' seja uma taxa de subnotificação definida no sistema
        subnotificacao_rate = self.subnotification_rate  # Exemplo baseado em estudos que indicam altas taxas de subnotificação
        idade_factor = 1 if agent.age < 50 else 1.5  # Aumenta a chance de progressão para estados mais graves em idades mais avançadas

        # print(f'txa de subnotificacao {subnotificacao_rate}')

        #print(f'agent status = {agent.status}')
        # print(f'probabilidade de progressao = {agent.prob_progressao}')


        # Transições de estado específicas do modelo SEPAI3R3O, ajustadas pela subnotificação e idade
        if agent.status == Status.Exposed:
            if np.random.rand() < agent.prob_progressao['E'] * subnotificacao_rate:
                agent.status = Status.PreSymptomatic
               

        elif agent.status == Status.PreSymptomatic:
            if np.random.rand() < agent.prob_progressao['P'] * idade_factor:
                agent.status = Status.Infected_Mild

        elif agent.status == Status.Asymptomatic:
            if np.random.rand() < agent.prob_progressao['A']:
                agent.status = Status.Recovered_Mild

        elif agent.status == Status.Infected_Mild:
            if np.random.rand() < agent.prob_progressao['I1'] * idade_factor:
                agent.status = Status.Infected_Severe
            else:
                agent.status = Status.Recovered_Severe

        elif agent.status == Status.Infected_Severe:
            if np.random.rand() < agent.prob_progressao['I2'] * idade_factor:
                agent.status = Status.Infected_Critical
            else:
                agent.status = Status.Recovered_Critical

        elif agent.status == Status.Infected_Critical:
            if np.random.rand() < agent.prob_morte * idade_factor:
                agent.status = Status.Death
            else:
                agent.status = Status.Recovered_Critical

        # Atualiza a riqueza do agente com base em suas despesas diárias mínimas
        agent.wealth -= self.minimum_expense * basic_income[agent.social_stratum]
        

    def execute(self):
        """
        Execute a complete iteration cycle of the Simulation, executing all actions for each agent
        in the population and updating the statistics
        """
        mov_triggers = [k for k in self.triggers_population if k['attribute'] == 'move']
        other_triggers = [k for k in self.triggers_population if k['attribute'] != 'move']

        for agent in self.population:
            self.move(agent, triggers=mov_triggers)
            self.update(agent)

            for trigger in other_triggers:
                if trigger['condition'](agent):
                    attr = trigger['attribute']
                    agent.__dict__[attr] = trigger['action'](agent.__dict__[attr])

        dist = np.zeros((self.population_size, self.population_size))

        contacts = []

        for i in np.arange(0, self.population_size):
            for j in np.arange(i + 1, self.population_size):
                ai = self.population[i]
                aj = self.population[j]

                if distance(ai, aj) <= self.contagion_distance:
                    contacts.append((i, j))
                    
        #print(f'disstancia contagio {contacts}')
        for par in contacts:
            ai = self.population[par[0]]
            aj = self.population[par[1]]
            self.contact(ai, aj)
            self.contact(aj, ai)

        if len(self.triggers_simulation) > 0:
            for trigger in self.triggers_simulation:
                if trigger['condition'](self):
                    attr = trigger['attribute']
                    self.__dict__[attr] = trigger['action'](self.__dict__[attr])

        self.statistics = None

    def get_positions(self):
        """Return the list of x,y positions for all agents"""
        return [[a.x, a.y] for a in self.population]

    def get_description(self, complete=False):
        """
        Return the list of Status and InfectionSeverity for all agents

        :param complete: a flag indicating if the list must contain the InfectionSeverity (complete=True)
        :return: a list of strings with the Status names
        """
        if complete:
            return [a.get_description() for a in self.population]
        else:
            return [a.status.name for a in self.population]

    def get_statistics(self, kind='info'):
        """
        Calculate and return the dictionary of the population statistics for the current iteration,
        considering the SEPAI3R3O model and possible underreporting.
        
        :param kind: 'info' for health statistics, 'ecom' for economic statistics, and None for all statistics
        :return: a dictionary
        """
        if self.statistics is None:
            self.statistics = {}
            for status in Status:
                self.statistics[status.value] = np.sum(
                    [1 for agent in self.population if agent.status == status]) / self.population_size
            
            #print(f'dentro de get_statistic = {self.statistics}')
            # Estimating underreporting based on symptomatic cases and testing rates
            estimated_underreporting_factor = 1  # This could be adjusted based on local health data
            self.statistics['Estimated_Active_Cases'] = (self.statistics['I1'] + self.statistics['I2'] + self.statistics['I3']) * estimated_underreporting_factor
            
            # Handling mortality separately due to its significance in the model
            self.statistics['Mortality'] = self.statistics['O'] / self.population_size

            # Adjusting recovered statistics to account for underreporting
            self.statistics['Estimated_Recoveries'] = (self.statistics['R1'] + self.statistics['R2'] + self.statistics['R3']) * estimated_underreporting_factor

            for quintile in [0, 1, 2, 3, 4]:
                self.statistics['Q{}'.format(quintile + 1)] = np.sum(
                    [a.wealth for a in self.population if a.social_stratum == quintile
                     and a.age >= 18 and a.status != Status.Death])
    
        return self.filter_stats(kind)


    def filter_stats(self, kind):
        if kind == 'info':
            return {k: v for k, v in self.statistics.items() if not k.startswith('Q') and k not in ('Business','Government')}
        elif kind == 'ecom':
            return {k: v for k, v in self.statistics.items() if k.startswith('Q') or k in ('Business','Government')}
        else:
            return self.statistics

    def __str__(self):
        return str(self.get_description())


class MultiPopulationSimulation(Simulation):
    def __init__(self, **kwargs):
        super(MultiPopulationSimulation, self).__init__(**kwargs)
        self.simulations = kwargs.get('simulations', [])
        self.positions = kwargs.get('positions', [])
        self.total_population = kwargs.get('total_population', 0)

    def get_population(self):
        population = []
        for simulation in self.simulations:
            population.extend(simulation.get_population())
        return population

    def append(self, simulation, position):
        self.simulations.append(simulation)
        self.positions.append(position)
        self.total_population += simulation.population_size

    def initialize(self):
        for simulation in self.simulations:
            simulation.initialize()

    def execute(self, **kwargs):
        for simulation in self.simulations:
            simulation.execute()

        for m in np.arange(0, len(self.simulations)):
            for n in np.arange(m + 1, len(self.simulations)):

                for i in np.arange(0, self.simulations[m].population_size):
                    ai = self.simulations[m].get_population()[i]

                    for j in np.arange(0, self.simulations[n].population_size):
                        aj = self.simulations[n].get_population()[j]

                        if np.sqrt(((ai.x + self.positions[m][0]) - (aj.x + self.positions[n][0])) ** 2 +
                                   ((ai.y + self.positions[m][1]) - (
                                           aj.y + self.positions[n][1])) ** 2) <= self.contagion_distance:
                            self.simulations[m].contact(ai, aj)
                            self.simulations[n].contact(aj, ai)
        self.statistics = None

    def get_positions(self):
        positions = []
        for ct, simulation in enumerate(self.simulations):
            for a in simulation.get_population():
                positions.append([a.x + self.positions[ct][0], a.y + self.positions[ct][1]])
        return positions

    def get_description(self, complete=False):
        situacoes = []
        for simulation in self.simulations:
            for a in simulation.get_population():
                if complete:
                    situacoes.append(a.get_description())
                else:
                    situacoes.append(a.status.name)

        return situacoes

    def get_statistics(self, kind='info'):
        """
        Calculate and return the dictionary of the population statistics for the current iteration,
        considering the SEPAI3R3O model and possible underreporting.
        
        :param kind: 'info' for health statistics, 'ecom' for economic statistics, and None for all statistics
        :return: a dictionary
        """
        if self.statistics is None:
            self.statistics = {}
            for status in Status:
                self.statistics[status.value] = np.sum(
                    [1 for agent in self.population if agent.status == status]) / self.population_size
            
            # Estimating underreporting based on symptomatic cases and testing rates
            estimated_underreporting_factor = 1  # This could be adjusted based on local health data
            self.statistics['Estimated_Active_Cases'] = (self.statistics['I1'] + self.statistics['I2'] + self.statistics['I3']) * estimated_underreporting_factor
            
            # Handling mortality separately due to its significance in the model
            self.statistics['Mortality'] = self.statistics['O'] / self.population_size

            # Adjusting recovered statistics to account for underreporting
            self.statistics['Estimated_Recoveries'] = (self.statistics['R1'] + self.statistics['R2'] + self.statistics['R3']) * estimated_underreporting_factor

        return self.filter_stats(kind)


    def __str__(self):
        return str(self.get_description())
