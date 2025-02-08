import re
import json

from tool import tool_template

prompt_template = """请尽可能准确地回答以下问题。你可以使用以下工具：

{tool_desc}

请使用以下格式：

Question: 你必须回答的输入问题  
Thought: 你应该始终思考下一步要做什么  
Action: 要执行的动作，必须是 [{tool_names}] 之一  
Action Input: 执行该动作所需的输入  
Observation: 执行该动作后的结果  
...（这个“Thought/Action/Action Input/Observation”循环可以重复零次或多次）  
Thought: 我现在知道最终答案了  
Final Answer: 对原始问题的最终答案  

开始吧！

Question: {prompt}  
Thought: """


class ReAct:
    def __init__(self, llm, tool_list, memory):
        # llm
        self.llm = llm

        # tool
        self.tool_list = tool_list
        self.tool_names = [t.name for t in self.tool_list]
        tool_desc = []
        for t in self.tool_list:
            tool_desc.append(
                tool_template.format(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                )
            )
        self.tool_desc = "\n".join(tool_desc)

        # memory
        with open(memory, "r", encoding="utf-8") as f:
            self.memory = json.load(f)
        self.messages = self.memory.copy()

    def detect_tool(self, text):
        # 定义正则表达式
        thought_pattern = r"Thought:\s*(.+)"
        action_pattern = r"Action:\s*(\w+)"
        action_input_pattern = r"Action Input:\s*(\{.*\}|\[.*\]|\".*\"|\'.*\')"

        # 使用正则匹配
        thought_match = re.search(thought_pattern, text)
        action_match = re.search(action_pattern, text)
        action_input_match = re.search(action_input_pattern, text)

        # 提取匹配结果
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        action_input = None

        if action_input_match:
            action_input_str = action_input_match.group(1).strip()
            try:
                action_input = json.loads(action_input_str)  # 解析 JSON
            except json.JSONDecodeError:
                action_input = action_input_str  # 解析失败，直接返回字符串

        return thought, action, action_input

    def call_tool(self, tool_name, tool_input):
        assert tool_name in self.tool_names, "工具不存在"
        for t in self.tool_list:
            if t.name == tool_name:
                try:
                    if isinstance(tool_input, str):
                        tool_input = json.loads(tool_input)

                    return t.call(tool_input)
                except:
                    return "工具参数错误"

    def call_llm(self, messages):
        response = self.llm.chat(messages)
        return response

    def call(self, prompt):
        prompt = prompt_template.format(
            tool_desc=self.tool_desc,
            tool_names=",".join(self.tool_names),
            prompt=prompt,
        )
        self.messages.append({"role": "user", "content": prompt})
        response = ""
        for _ in range(5):
            output = self.call_llm(self.messages)
            thought, action, action_input = self.detect_tool(output)
            if action:
                observation = self.call_tool(action, action_input)
                taaio = f"{thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}\nThought: "
                response += taaio
                # taaio添加到user的消息中，来让llm继续生成
                self.messages[-1]["content"] += taaio
            else:
                response += output
                # output添加到assistant的消息中
                # why output not response: response里有一部分已经在user的消息里了，如果是response，重复内容过多
                self.messages.append({"role": "assistant", "content": output})
                break

        return "Thought: " + response
