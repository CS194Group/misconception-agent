
# TODO
# Implement tools 
# import logging
import dspy
from openai import OpenAI
import os
import json
import time 
import httpx
import pdb
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

class Calculator:
    def __init__(self):
        self.name = "Calculator"
    
    def __call__(self, QuestionText, CorrectAnswer):
        try:
            return {"result": str(eval(QuestionText))}
        except Exception as e:
            return {"error": str(e)}

class WebSearchTool:
    def __init__(self):
        self.name = "WebSearch"
    
    def __call__(self, QuestionText, CorrectAnswer):
        return {"result": f"Search results for: {QuestionText}"}

class WikipediaSearchTool:
    def __init__(self):
        self.name = "WikipediaSearch"
    
    def __call__(self, QuestionText, CorrectAnswer):
        return {"result": f"Wikipedia summary for: {QuestionText}"}


tools_basic = {
    "calculator": Calculator(),
    "websearch": WebSearchTool(),
    "wikipediasearch": WikipediaSearchTool()
}

# This agent is use to solve the problem
class SelectAgentSignature(dspy.Signature):
    """Choose only one tool from the provided options that is the most helpful in solving the problem."""
    tool_list = dspy.InputField(desc="Tool list to choose from: You can only select from these tools.")
    context = dspy.InputField(
        desc='Thought history of other agents for reference, you need to judge whether you are able to answer the question based on these history. Empty if no history is available.')
    QuestionText = dspy.InputField(desc='The question text.')
    ConstructName = dspy.InputField()
    SubjectName = dspy.InputField(desc="The subject of the question.")
    CorrectAnswer = dspy.InputField(desc="The correct answer.")
    Choice = dspy.OutputField(desc="Only output the name of tool you select without explanation.")

class SolveAgentSignature(dspy.Signature):
    """Can I get the correct answer of this question based on the context I have? If so, output 'Yes" without any further explanation, if not, answer just answer "No" without any explanation """
    context = dspy.InputField(
        desc='Thought history of other agents for reference, you need to judge whether you are able to answer the question based on these history. Empty if no history is available.')
    QuestionText = dspy.InputField(desc='The question text.')
    ConstructName = dspy.InputField()
    SubjectName = dspy.InputField(desc="The subject of the question.")
    CorrectAnswer = dspy.InputField(desc="The correct answer.")
    Judge = dspy.OutputField(desc= "If you have sufficient infomation to answer the question, output 'Yes'. Else output 'No'. Only output 'Yes' or 'No' without any explanation") 
    

class SummaryAgentSignature(dspy.Signature):
    """Based on the provided context (thought history) and the correct answer, summarize the final reasoning process and answer in an organized, step-by-step manner."""
    context = dspy.InputField(
        desc='Thought history of other agents for reference, you need to judge whether you are able to answer the question based on these history. Empty if no history is available.')
    QuestionText = dspy.InputField(desc='The question text.')
    ConstructName = dspy.InputField()
    SubjectName = dspy.InputField(desc="The subject of the question.")
    CorrectAnswer = dspy.InputField(desc="The correct answer.")
    Solution = dspy.OutputField(desc="Well organized rational final solution.")

class SolveAgent(dspy.Module):
    def __init__(self, name, tools=tools_basic, persona_promt=None):
        super().__init__()
        self.name = name
        self.tools = tools
        if not self.tools:
            self.tools = tools_basic
        self.prefix_promt = persona_promt

        self.utils_agent = dspy.Predict(SelectAgentSignature)
        self.solve_agent = dspy.Predict(SolveAgentSignature)
        self.summery_agent = dspy.Predict(SummaryAgentSignature)

    def get_tool_description(self, tool):

        descriptions = {
            "Calculator": "Mathematical calculations and numerical operations",
            "WebSearch": "Real-time search and access to online information",
            "WikipediaSearch": "Search for encyclopedia knowledge and academic information"
        }
        return descriptions.get(tool.name, "General tool")

    def forward(self, QuestionText, ConstructName, SubjectName, CorrectAnswer, context=None) -> str:
        # Directly pass the inputs to the process method
        try:
            while(True):
                tool_descriptions = "\n".join([
                    f"{key}: Suitable for {self.get_tool_description(tool)}"
                    for key, tool in self.tools.items()
                ])

                # Aelect the relavant tool
                tool_selection = self.utils_agent(
                    tool_list=f"tool list:\n{tool_descriptions}",
                    context=context,
                    QuestionText=QuestionText,
                    ConstructName=ConstructName,
                    SubjectName=SubjectName,
                    CorrectAnswer=CorrectAnswer,
                    prefix = self.prefix_promt
                )

                # Add the selection to history
                # logging.info(f"Selction is: {tool_selection}")

                if context:
                    context = str(context) +  f"\nTool selection: {tool_selection} \n"
                else:
                    context = f"\nTool selection: {tool_selection} \n"

                # Call relavent tool
                tool_selection = tool_selection.completions[0].Choice.lower()
                matched_tool = self.tools.get(tool_selection, WebSearchTool())
                thoughts = matched_tool(QuestionText, CorrectAnswer)
                context += f"\nTool use result: {thoughts} \n"
                # logging.info(f"Tool use result is: {thoughts}")

                # Judge whether the infomation is enough
                judge_pass = self.solve_agent(
                    context=context,
                    QuestionText=QuestionText,
                    ConstructName=ConstructName,
                    SubjectName=SubjectName,
                    CorrectAnswer=CorrectAnswer,
                    prefix = self.prefix_promt
                )

                judge_pass = judge_pass.completions[0].Judge.lower()
                if 'yes' in judge_pass:
                    break
                    # logging.debug(f"Current judge is: {judge_pass}")

            outputs = self.summery_agent(
                context=context,
                QuestionText=QuestionText,
                ConstructName=ConstructName,
                SubjectName=SubjectName,
                CorrectAnswer=CorrectAnswer,
                prefix = self.prefix_promt
            )

            return outputs.completions[0].Solution
        except Exception as e:
            # print(e)
            return "Failed to generate anwser explanation of the problem."
        
MisagentPrompt = r"""
Based on the provided correct answer, its reasoning process, and the incorrect answer obtained, identify the step where the error occurred, and determine the reason for the mistake.
"""

# MisagentPrompt = r"""
# Based on the provided correct answer, its reasoning process, and the incorrect answer obtained, identify the step where the error occurred, and determine the reason for the mistake.
# You should carefully examine each step for potential errors and identify the step most likely to have mistakes. You need to consider, but are not limited to, the following situations:  

# 1. Calculation Errors:  
#    For example, when calculating \( 8 \times 7 - 3 \), it is mistakenly written as  
#    \[
#    8 \times 7 - 3 = 56 - 3 = 53,
#    \]  
#    while the correct answer is actually  
#    \[
#    8 \times 7 - 3 = 56 - 3 = 53.
#    \]  
#    This example highlights the importance of operation order and calculation details.  

# 2. Conceptual Understanding Errors:  
#    Failing to distinguish that converting percentages to fractions requires dividing by 100, not 10.  
#    For instance, \( 75\% \) is incorrectly converted to \( 75/10 \), while the correct conversion should be  
#    \[
#    75\% = \frac{75}{100} = \frac{3}{4}.
#    \]  

# 3. Logical Reasoning Errors:  
#    Mistakenly reasoning in reverse when proving "if \( A \) is true, then \( B \) must also be true," as "if \( B \) is true, then \( A \) must also be true."  
#    For example, in the statement "if it is a mammal, then it is warm-blooded," it cannot be deduced that "all warm-blooded animals are mammals," because birds are also warm-blooded animals.  

# 4. Errors in Visualization and Spatial Imagination:  
#    When calculating the surface area of a cube, mistakenly adding up all areas without accounting for overlapping faces.  
#    For example, for a cube with an edge length of 2, the surface area is incorrectly calculated as  
#    \[
#    2 \times 6 = 12,
#    \]  
#    while the correct answer is  
#    \[
#    2^2 \times 6 = 24.
#    \]  

# 5. Algebraic Expression Errors:  
#    When solving a system of equations, for example, \( x + y = 10 \) and \( x - y = 2 \), mistakenly adding the two equations to get  
#    \[
#    2x = 12,
#    \]  
#    and directly concluding \( x = 6 \). This method ignores that the solution must satisfy both equations simultaneously.  

# Your output format should be:  
# \[
# \text{[reasoning step index]: Explanation of the potential error.}
# \]
# """

# This agent is use to find the most possible wrong process of the question and get the misconception
class MisAgentSignature(dspy.Signature):
    """Explain the misconception the student has based on correct answer reasoning and his answer."""
    context = dspy.InputField(
        desc='Debate history of other agents for reference. Empty if no history is available.')
    QuestionText = dspy.InputField(desc='The question text.')
    AnswerText = dspy.InputField(desc='The student wrong answer text.')
    ConstructName = dspy.InputField()
    SubjectName = dspy.InputField(desc="The subject of the question.")
    CorrectAnswer = dspy.InputField(desc="The correct answer.")
    CorrectReasoning = dspy.InputField(desc="The correct answer' reasoning process.")
    MisconceptionText = dspy.OutputField(desc=MisagentPrompt )
    
class MisAgent(dspy.Module):
    def __init__(self, name, persona_promt=None):
        super().__init__()
        self.name = name
        self.prefix_promt = persona_promt
        self.process = dspy.Predict(MisAgentSignature)

    def forward(self, QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, CorrectReasoning, context=None) -> str:
        # Directly pass the inputs to the process method
        try:
            outputs = self.process(
                context=context,
                QuestionText=QuestionText,
                AnswerText=AnswerText,
                ConstructName=ConstructName,
                SubjectName=SubjectName,
                CorrectAnswer=CorrectAnswer,
                CorrectReasoning=CorrectReasoning,
                prefix = self.prefix_promt
            )

            return outputs.completions[0].MisconceptionText
        except Exception as e:
            return "Failed to generate misconception explanation."
        
# This agent is use to summarize the misconception
class FinAgentSignature(dspy.Signature):
    """Summarize the final misconception based on the correct reasoning process and the possible errors in the incorrect answer."""
    context = dspy.InputField(
        desc='Debate history of other agents for reference. Empty if no history is available.')
    QuestionText = dspy.InputField(desc='The question text.')
    AnswerText = dspy.InputField(desc='The student wrong answer text.')
    ConstructName = dspy.InputField()
    SubjectName = dspy.InputField(desc="The subject of the question.")
    CorrectAnswer = dspy.InputField(desc="The correct answer.")
    CorrectReasoning = dspy.InputField(desc="The correct answer' reasoning process.")
    MisconceptionReasoning = dspy.InputField(desc="The correct answer' reasoning process.")

    MisconceptionText = dspy.OutputField(
        desc='Based on CorrectReasoning and MisconceptionReasoning, summarize and extract the misconception of the incorrect answer, and output only the misconception without any reasoning.')
    
class FinAgent(dspy.Module):
    def __init__(self, name, persona_promt=None):
        super().__init__()
        self.name = name
        self.prefix_promt = persona_promt
        self.process = dspy.Predict(FinAgentSignature)

    def forward(self, QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, CorrectReasoning, MisconceptionReasoning, context=None) -> str:
        # Directly pass the inputs to the process method
        try:
            outputs = self.process(
                context=context,
                QuestionText=QuestionText,
                AnswerText=AnswerText,
                ConstructName=ConstructName,
                SubjectName=SubjectName,
                CorrectAnswer=CorrectAnswer,
                CorrectReasoning=CorrectReasoning,
                MisconceptionReasoning = MisconceptionReasoning,
                prefix = self.prefix_promt
            )

            return outputs.completions[0].MisconceptionText
        
        except Exception as e:
            # print(e)
            return "Failed to generate misconception explanation."
        
class SolveAgent_api(dspy.Module):

    def __init__(self, name, persona_promt=None, request_interval=1):
        super().__init__()
        self.name = name
        self.prefix_promt = persona_promt
        self.request_interval = request_interval

        self.solve_agent = dspy.Predict(SolveAgentSignature)
        self.summery_agent = dspy.Predict(SummaryAgentSignature)

    def get_reasoning(self, query, answer):

        try:
            # pdb.set_trace()
            prompt = f"Please generate proper reasoning process of the question.\nQuestion:\n{query}\nCorrect Answer:{answer}. Your answer should be well-formatted, using 1. 2. 3. to list items sequentially."

            http_client = httpx.Client(verify=False)

            time.sleep(self.request_interval)
            client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                http_client=http_client
            )
            completion = client.chat.completions.create(
                model="qwen2-math-72b-instruct",
                # model="qwen-math-plus",
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant that giit.'},
                    {'role': 'user', 'content': prompt}]
                )
            # print(completion.model_dump_json())
            return completion.choices[0].message.content
        
        except Exception as e:
            print(e)
            # pdb.set_trace()

    def forward(self, QuestionText, ConstructName, SubjectName, CorrectAnswer, context=None) -> str:
        # Directly pass the inputs to the process method
        # try:
        while(True):
            
            thoughts = self.get_reasoning(QuestionText, CorrectAnswer)

            if context:
                context += f"\Reasoning result: {thoughts} \n"
            else:
                context = f"\Reasoning result: {thoughts} \n"

            # Judge whether the infomation is enough
            judge_pass = self.solve_agent(
                context=context,
                QuestionText=QuestionText,
                ConstructName=ConstructName,
                SubjectName=SubjectName,
                CorrectAnswer=CorrectAnswer,
                prefix = self.prefix_promt
            )

            judge_pass = judge_pass.completions[0].Judge.lower()
            if 'yes' in judge_pass:
                break

        outputs = self.summery_agent(
            context=context,
            QuestionText=QuestionText,
            ConstructName=ConstructName,
            SubjectName=SubjectName,
            CorrectAnswer=CorrectAnswer,
            prefix = self.prefix_promt
        )

        return outputs.completions[0].Solution
        # except Exception as e:
        #     return "Failed to generate anwser explanation of the problem."