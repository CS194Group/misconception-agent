import time
import pdb
from typing import Literal

import dspy

#########################################################################################################################
# The main model (ultilizing all agents together)


class SharedMemoryPool:
    def __init__(self):
        self.memories = []

    def add_memory(self, memory, agent_id):
        self.memories.append({
            'content': memory,
            'agent_id': agent_id,
            'timestamp': len(self.memories)
        })

    def get_relevant_memories(self, k=5):
        return self.memories[-k:] if len(self.memories) > k else self.memories


class ExchangeOfThought(dspy.Module):
    # def __init__(self, agent_a, agent_b, agent_c, rounds: int = 1, mode: Literal["Debate", "Report", "Memory", "Relay"] = "Report"):
    def __init__(self, agent_a, agent_b, agent_c, agent_d=None, agent_e=None, rounds: int = 1, mode: Literal["Debate", "Report", "Memory", "Relay"] = "Report"):
        super().__init__()
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.agent_c = agent_c
        self.agent_d = agent_d
        self.agent_e = agent_e
        self.memory_pool = SharedMemoryPool()
        self.rounds = rounds
        self.mode = mode
        # assert self.mode == "Report"

        # # below is the multi init
        # self.expert_group_a = expert_a
        # self.expert_group_b = expert_b
        # self.expert_group_c = expert_c

    def forward(self, QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer):
        #time.sleep(0.5)
        if self.mode == "Report":
            return self._report_mode(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer)
        elif self.mode == "Debate":
            return self._debate_mode(QuestionText)
        elif self.mode == "Memory":
            return self._memory_mode(QuestionText)
        elif self.mode == "Relay":
            return self._relay_mode(QuestionText)
        elif self.mode == "multi":
            return self._multi_mode(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer)
        elif self.mode == "multi_4":
            return self._multi4_mode(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer)
        elif self.mode == "bigram":
            return self._bigram_mode(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _report_mode(self, QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer):
        # Step 1: A initiates thought
        thought_a = self.agent_a(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer)
        # pdb.set_trace()

        # Note this for-loop does not keep history of previous rounds, but it includes the chain of toughts if the agents
        for _ in range(self.rounds):
            # Step 2: A sends thought to B and C
            agent_a_history = f"Agent A concludes: ({str(thought_a)})"
            thought_b = self.agent_b(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, context=agent_a_history)
            thought_c = self.agent_c(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, context=agent_a_history)

            # Step 3: A receives feedback from B and C, then combines thoughts
            combined_thoughts = (f"Agent B concludes: ({str(thought_b)}) /n"
                                 f"Agent C concludes:  ({str(thought_c)})")
            thought_a = self.agent_a(
                QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, context=combined_thoughts)

        return thought_a

    def _debate_mode(self, question):
        # Step 1: B and C initiate thought
        thought_b = self.agent_b.forward(question)
        thought_c = self.agent_c.forward(question)

        for _ in range(self.rounds):
            # Step 2: B and C communicates back and forth
            thought_b = self.agent_b.forward(question, context=thought_c)
            thought_c = self.agent_c.forward(question, context=thought_b)

        # Step 3: B and C send their final thoughts to A
        combined_thoughts = f"Agent B concludes: ({thought_b}), Agent C concludes: ({thought_c})"
        thought_a = self.agent_a.forward(question, context=combined_thoughts)
        thought_a.question = question

        return thought_a

    def _memory_mode(self, question):
        thought_a = self.agent_a.forward(question)
        thought_b = self.agent_b.forward(question)
        thought_c = self.agent_c.forward(question)
        self.memory_pool.add_memory(thought_a, 'Agent_a')
        self.memory_pool.add_memory(thought_b, 'Agent_b')
        self.memory_pool.add_memory(thought_c, 'Agent_c')

        for _ in range(self.rounds - 1):
            self.memory_pool.add_memory(self.agent_a.forward(
                question,
                context=self.memory_pool.get_relevant_memories()
            ), 'Agent_a')

            self.memory_pool.add_memory(self.agent_b.forward(
                question,
                context=self.memory_pool.get_relevant_memories()
            ), 'Agent_b')

            self.memory_pool.add_memory(self.agent_c.forward(
                question,
                context=self.memory_pool.get_relevant_memories()
            ), 'Agent_c')

        thought_a = self.agent_a.forward(
            question,
            context=self.memory_pool.get_relevant_memories(k=100)
        )

        thought_a.question = question

        return thought_a

    def _relay_mode(self, question):
        thought_a = self.agent_a.forward(question)

        for _ in range(self.rounds):
            thought_b = self.agent_b.forward(question, context=thought_a)
            thought_c = self.agent_b.forward(question, context=thought_b)
            thought_a = self.agent_b.forward(question, context=thought_c)

        thought_a.question = question

        return thought_a
    
    def _multi4_mode(self, QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer):
        # Step 1: A initiates thought
        thought_a = self.agent_a(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer)


        for _ in range(self.rounds):

            agent_a_history = f"Agent A concludes: ({str(thought_a)})"
            thought_b = self.agent_b(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, context=agent_a_history)
            thought_c = self.agent_c(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, context=agent_a_history)

            prompt = f"For this question's misconception, student b's ideas is \n{thought_b}\nstudent c's ideas is \n{thought_c}\n"

            thought_d = self.agent_d.forward(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, context=prompt)

            thought_a = self.agent_a.forward(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, context=thought_d)

        return thought_a
    
    def _bigram_mode(self, QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer):
        # Step 1: A initiates thought
        thought_a = self.agent_a(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer)

        for _ in range(self.rounds):

            thought_b = self.agent_b.forward(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, context=thought_a)
            thought_c = self.agent_c.forward(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, context=thought_a)

            thought_b = self.agent_b.forward(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, context=thought_c)
            thought_c = self.agent_c.forward(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, context=thought_b)

            prompt = f"For this question's misconception, student b's ideas is \n{thought_b}\nstudent c's ideas is \n{thought_c}\n"

            thought_a = self.agent_a.forward(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, context=prompt)

        return thought_a


#########################################################################################################################
