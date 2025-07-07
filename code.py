import json5
import os
import json
import re
import random
from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print
from qwen_agent.tools.base import BaseTool, register_tool
from openai import OpenAI
import time
from collections import Counter

openai_client = OpenAI(
    
    api_key=""
)


DEFAULT_AGENT_WEIGHTS = {
    'bot1': 1.0, 'bot2': 1.2, 'bot3': 1.1,
    'bot4': 1.0, 'bot5': 1.2, 'bot6': 1.1, # PPTL生成Agent
    'bot7': 1.0,                          # GPT-4o
}

DEFAULT_AGENT_PERFORMANCE = {
    f'bot{i}': {'correct': 0, 'total': 0, 'calls': 0} for i in range(1, 8)
}


agent_weights = {}
agent_performance = {}


system_instruction_reflect_propositions = '''
你是一位逻辑学家和语言哲学家，是形式化方法领域的最终权威。你的任务是担任“命题审查守门员”，以最严苛的标准，验证一组从自然语言句子中提取的命题（Propositions）是否达到了形式化前的“绝对纯净”状态。
你的判断是后续所有逻辑转换工作的基础，绝不容许任何瑕疵。

--- 核心原则：命题的本质 ---
命题是逻辑的“原子”，是不可再分的、能够被赋予真/假值的最小陈述单元。在审查时，你必须坚守以下三大原则：

1.  **原子性 (Atomicity - 必须是最小单元)**:
    - 命题必须代表一个单一、独立的事件或状态。
    - 任何包含逻辑关系（如因果、时序、条件）的陈述都必须被拆分。
    - 案例：`"If A happens, then B will occur"` 必须拆分为两个命题: `P: A happens` 和 `Q: B will occur`。`"A then B"` 也是同理。
    - 案例：`"A happens after B is completed"` 必须拆分为: `P: A happens` 和 `Q: B is completed`。“after”是时序关系，属于PPTL公式的范畴，绝不能出现在命题内部。
    - 案例: `Person is not available` 应该拆分为`Person is available` not这种表否定的需要忽略
2.  **事实纯净性 (Factual Purity - 只陈述事实，不解释关系)**:
    - 命题只描述“什么（what）”发生了，而不描述“如何（how）”、“何时（when）”或“为何（why）”。
    - 它必须剥离所有逻辑连词、时序连词和条件连词。这些连接关系将在下一步的PPTL公式构建中通过 `=>`, `&&`, `||`, `○`, `□` 等操作符来表达。
    - **你的任务就是确保原材料（命题）的纯净，以便工匠（PPTL生成器）能够正确地使用工具（逻辑操作符）。**

3.  **完整性与精确性 (Completeness & Fidelity)**:
    - **完整性**: 原始句子中的每一个独立的条件、事件或状态都必须被提取出来，无一遗漏。
    - **精确性**: 命题的措辞必须严格忠于原文，**禁止任何形式的释义、概括或简化**。原文的每个关键词（如 `immediately`, `completely`, `requested`）都至关重要，必须保留。

--- 禁忌事项：你绝对不能做的事 ---
-   **禁止思考PPTL**: 在你的世界里，`○`, `□`, `W`, `U` 等时序逻辑符号根本不存在。你的任务发生在这一切之前。
-   **禁止保留逻辑/时序词**: 命题中绝不应出现 `if`, `then`, `and`, `or`, `after`, `before`, `until`, `unless`, `whenever` 等词语。这些是构建公式的“粘合剂”，不是“砖块”（命题）的一部分。

--- 审查流程与案例分析 ---

**--- 正确的案例 (学习应该怎么做) ---**
案例1 (条件与结果):
原始语句: "Whenever the button to some floor is pushed, the elevator will reach there."
正确的命题:
P: the button to some floor is pushed
Q: the elevator will reach there
*审查意见: 正确。完整地提取了条件和结果，且命题是原子的、纯净的。*

案例2 (复合条件):
原始语句: "If the door is opened and the alarm is armed, the siren will sound immediately."
正确的命题:
P: the door is opened
Q: the alarm is armed
R: the siren sounds immediately
*审查意见: 正确。将'and'连接的两个条件拆分为独立的原子命题P和Q，结果R也准确包含了关键词'immediately'。*

**--- 错误案例分析 (学习什么不该做) ---**
案例3 (时序污染):
原始语句: "An acknowledgment is sent after the message is received."
**错误的命题**:
P: An acknowledgment is sent after the message is received.
*审查意见: **错误**。违反了[原子性]和[事实纯净性]原则。命题P包含了时序关系'after'，它不是一个原子事实。*
**正确的命题**:
P: An acknowledgment is sent
Q: the message is received

案例4 (条件污染):
原始语句: "If the system is in auto mode, it monitors the pressure."
**错误的命题**:
P: the system is in auto mode and monitors the pressure
*审查意见: **错误**。违反了[原子性]。这个命题混合了条件和结果，没有将它们拆分开。*
**正确的命题**:
P: the system is in auto mode
Q: it monitors the pressure

--- 你的输出 ---
你必须以严格的中文JSON格式返回你的审查结果。

JSON格式:
{
  "is_correct": true/false,
  "feedback": "如果 is_correct 为 false，请用中文并引用上述[原则]来精确地指出问题所在。例如：'不正确。违反了[原子性]原则，命题P “A and B” 必须被拆分为两个独立的命题。' 或 '不正确。违反了[事实纯净性]原则，命题Q包含了时序词“after”，应将其移除并创建独立的命题。' 如果 is_correct 为 true，用中文简要说明其如何满足了核心原则。"
}
'''


system_instruction_reflect_pptl = '''你是一位精通形式化方法和时序逻辑的专家审查官。你的唯一任务是：验证一个给定的PPTL公式是否正确地形式化了一个自然语言陈述，并且与提供的命题完全对应。

--- 前提 ---
- 在此阶段，你必须**假设输入的命题列表是完全正确的**。你的工作不包括审查命题本身。

--- PPTL 核心语法速查表 (审查依据) ---
**时序操作符 (Temporal Operators): 这是审查的重中之重！**
- `○`: **下一刻 (Next)**。用于表达 "then next",当句子中含有next或者时态是表达当前时，则你的pptl应该含有此符号。应该是`P => ○Q` 而 `P => Q` 表达将来的某一时刻！
- `□`: **总是 (Always / Globally)**。审查要点: 规则是否是全局约束？(如 "whenever", "always", 或隐含的全局规则)。
- `○^n`: **n个时刻之后**。审查要点: 是否有明确的时间延迟？ (如 "in 3 seconds", "after 5 cycles"等)。
- `你需要注意原始语句中是否有not这类词，如果有并且提取的命题中没有not，那么你需要在命题前加'!'这个符号`
- **常见错误**: 绝对要区分 `P => Q` (将来某一时刻发生) 和 `P => ○Q` (下一刻发生)！
- 所有规则通常都应被`□()`包裹，表示全局成立,除非自然语言中有表达是最终发生意思比如说finally这种单词或者
其它的表示最终发生(比如it is going to )或者一定会发生的含义时那么你需要将`□()`替换为`◇()`,在添加`◇()`你需要分辩是在整个PPTL添加还是在其中的某一个命题添加下面这个例子你可以进行参考:
NL:after UHtoxEw, it will happen that hSHRy
PPTL:□(( UHtoxEw ) -> ◇( hSHRy ))

NL:at some point both rjZKDoeHuU and CCrIMvt
PPTL:◇( rjZKDoeHuU && CCrIMvt )
--- 你的审查任务 ---
1.  **逻辑结构**: `&&`, `||`, `=>` 等逻辑连接词是否正确反映了句子的结构？(如 "and", "or", "if...then")。
2.  **时序准确性**: `○`, `□` 等时序操作符的使用是否精确无误？
3.  **变量一致性**: 公式中使用的命题字母（P, Q, ...）是否与提供的命题列表完全对应？
4.  **中文JSON输出**: 以指定的中文JSON格式返回你的审查结果。

--- 正确的案例供你参考进行学习---
案例1 (未来发生):
自然语言: "Whenever the button to some floor is pushed, the elevator will reach there."
命题:
P: the button to some floor is pushed
Q: the elevator will reach there
正确的PPTL: □(P => Q)
理由: "Whenever" 对应 `□`，"will reach" 暗示未来发生，但不确定是哪一时刻。

案例2 (立即发生):
自然语言: "If the door is opened and the alarm is armed,next the siren will sound immediately."
命题:
P: the door is opened
Q: the alarm is armed
R: the siren sounds
正确的PPTL: □((P && Q) => ○R)
理由: "next" 表示下一时刻发生，所以使用 `○`。

案例3 (延时发生):
自然语言: "If the 'Air_Ok' signal remains low for 3 seconds, the auto control mode is terminated."
命题:
P: 'Air_Ok' signal is low
Q: auto_control_mode is terminated
正确的PPTL: □((P && ○P && ○^2 P) => ○^3 Q)
理由: "remains low for 3 seconds" (持续3秒)被精确地形式化为连续三个时刻P为真，结果在第三个时刻之后发生。

案例4 (含有NOT):
自然语言: If i2 is pushed, i2 will not be popped, before floor2 is reached.
P: 'i2 is pushed'
Q: 'i2 will  be popped'
R: 'floor2 is reached'
PPTL : ' □(P → (!Q U R) )'
JSON格式:
{
  "is_correct": true/false,
  "feedback": "如果 is_correct 为 false，请用中文详细指出错误，例如：'不正确。原始陈述“will be sent”暗示了状态转变，必须使用'○'操作符。当前公式缺少'○'。' 如果 is_correct 为 true，用中文简要说明其逻辑和时序的正确性。"
}
'''

system_instruction_generate = '''你是一个为PPTL（命题投影时序逻辑）从英文句子中提取命题条件的、具备学习能力的机器人。

--- 学习与修正指南 ---
在某些情况下，你可能会收到来自高级审查员的“重要修正指南”。这表示你之前的尝试存在问题。
当你看到这个指南时，你必须：
1.  **优先处理反馈**: 将修正指南作为最高优先级的指令。
2.  **理解错误**: 仔细阅读指南中指出的问题所在。
3.  **精确修正**: 在本次生成中，确保你的输出完全解决了指南中提到的所有问题。
4.  **关注细节**: 如果句子中的命题含有not你提取的命题应该不含not
这是一个从错误中学习并提升你能力的机会。

--- 核心任务 ---
给定一个英文句子，你必须:
1.  只返回命题条件，每行一个，格式为 'P: ...', 'Q: ...', 直到 'Z: ...', 不带任何额外信息。
2.  不要调用任何工具。
3.  确保命题条件准确反映句子的逻辑关系。


--- 正确的案例供你参考进行学习---
英文句子: "Whenever the button to some floor is pushed, the elevator will reach there."
命题条件:
P: the button to some floor is pushed
Q: the elevator will reach there

英文句子: "If the door is opened and the alarm is armed, the siren will sound immediately."
命题条件:
P: the door is opened
Q: the alarm is armed
R: the siren sounds

英文句子: "If telephonic_info or ordering is selected, text_data_information is displayed."
命题条件:
P: telephonic_info is selected
Q: ordering is selected
R: text_data_information is displayed


英文句子: If i2 is pushed, i2 will not be popped, before floor2 is reached.
P: 'i2 is pushed'
Q: 'i2 will  be popped'
R: 'floor2 is reached'

'''


system_instruction_debate = '''你是一个参与辩论以评估命题条件的机器人,。你将收到:
- 原始的英文句子。
- 三个带编号的命题输出。
- 在后续轮次中，其他Agent的投票和理由。
你的任务:
1.  分析三个输出，并选择最好的带编号输出（1, 2, 或 3）。
2.  以以下格式提供你的投票和理由:
    Vote: <数字>
    Reason: <文本>
3.  选择标准: 逻辑一致性、完整性、时序逻辑正确性。
4.  你需要说中文
'''

# [反馈学习增强] Qwen PPTL生成指令
system_instruction_pptl = '''你是一个精通时序逻辑的专家，负责将自然语言陈述和命题转换为PPTL公式，并能从反馈中学习。

--- 学习与修正指南 ---
在某些情况下，你可能会收到来自高级审查员的“重要修正指南”。这表示你之前生成的PPTL公式存在逻辑或语法错误。
当你看到这个指南时，你必须：
1.  **优先处理反馈**: 将修正指南作为最高优先级的指令。
2.  **定位错误**: 仔细阅读指南，理解你的公式错在哪里（例如，是缺少'○'，还是逻辑连接词用错）。
3.  **精确修正**: 对照下面的“PPTL核心语法速查表”，生成一个解决了所有已指出问题的、完全正确的PPTL公式。
4.你需要格外注意命题是不是含有not,如果原始语句的某个命题含有not但是提取出来的命题里没有not，你在转换成Pptl时需要加上!，比如!q这样。
这是一个提升你逻辑推理能力的机会。

--- PPTL 核心语法速查表 ---
**基本命题:** P, Q, R, ... (由输入提供)
**逻辑连接词:** `&&` (与), `||` (或), `!` (非), `=>` (蕴含)
**时序操作符 (Temporal Operators):**
- `○`: **下一刻 (Next)**。用于表达 "then next",当句子中含有next或者时态是表达当前时，则你的pptl应该含有此符号。应该是`P => ○Q` 而 `P => Q` 表达将来的某一时刻！
- `□`: **总是 (Always / Globally)**。用于包裹全局规则。
- `○^n`: **n个时刻之后**。用于表达明确延迟，如 "in 3 seconds"。
- 所有规则通常都应被`□()`包裹，表示全局成立,除非自然语言中有表达是最终发生意思比如说finally这种单词或者
其它的表示最终发生(比如it is going to )或者一定会发生的含义时那么你需要将`□()`替换为`◇()`,在添加`◇()`你需要分辩是在整个PPTL添加还是在其中的某一个命题添加下面这个例子你可以进行参考:

NL:after UHtoxEw, it will happen that hSHRy
PPTL:□(( UHtoxEw ) -> ◇( hSHRy ))

NL:at some point both rjZKDoeHuU and CCrIMvt
PPTL:◇( rjZKDoeHuU && CCrIMvt )
----标准答案供你学习----
案例1 (未来发生):
自然语言: "Whenever the button to some floor is pushed, the elevator will reach there."
命题:
P: the button to some floor is pushed
Q: the elevator will reach there
PPTL: □(P => Q)

案例2 (立即发生):
自然语言: "If the door is opened and the alarm is armed, next the siren will sound immediately."
命题:
P: the door is opened
Q: the alarm is armed
R: the siren sounds
PPTL: □((P && Q) => ○R)

案例3 (延时发生):
自然语言: "If the 'Air_Ok' signal remains low for 3 seconds, the auto control mode is terminated."
命题:
P: 'Air_Ok' signal is low
Q: auto_control_mode is terminated
PPTL: □((P && ○P && ○^2 P) => ○^3 Q)

案例4(含有not)
自然语言: If i2 is pushed, i2 will not be popped, before floor2 is reached.
命题:
P: 'i2 is pushed'
Q: 'i2 will be popped'
R: 'floor2 is reached'
PPTL: □(P => □(!Q U R))

--- 核心任务 ---
1.  **深入分析**: 仔细阅读自然语言陈述，特别注意时序关键词（如 "whenever", "immediately", "will be", "for X seconds"）。
2.  **精确转换**: 根据速查表和任何修正指南，将陈述精确地转换为PPTL公式。
3.  **纯净输出**: 只返回最终的PPTL公式，不含 "PPTL:" 等前缀。
'''

# Qwen PPTL辩论指令
system_instruction_pptl_debate = '''你是一个参与PPTL公式评估的辩论专家。

--- PPTL 核心语法速查表 ---
**时序操作符:** `○` (Next), `□` (Always), `W` (Weak Until / before)
 - **特别注意**: `P => Q` (立即蕴含) 和 `P => ○Q` (下一刻蕴含) 是完全不同的！请仔细检查自然语言中的时序关系。

--- 你的任务 ---
1.  **严格审查**: 使用上述速查表，对照原始陈述，逐一审查每个方案的语法和逻辑正确性。
2.  **选出最佳**: 找出最能精确表达原始陈述逻辑的方案。
3.  **提供投票**: 以以下格式提供你的投票和详细理由:
    Vote: <数字>
    Reason: <用中文详细解释你选择的理由，并可以指出其他方案的错误。>
'''



def load_knowledge_base(file_path: str) -> list:
    """
    从指定文件加载人工校对的知识库。
    文件格式应为：
    自然语言语句1
    P: 命题1
    Q: 命题2
    ...

    自然语言语句2
    P: 命题A
    ...
    """
    if not os.path.exists(file_path):
        print(f"知识库文件 '{file_path}' 未找到，将不加载额外知识。")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    knowledge_base = []
    entries = content.strip().split('\n\n')
    for entry in entries:
        if not entry.strip():
            continue
        lines = entry.strip().split('\n')
        if len(lines) < 2:
            continue
        sentence = lines[0]
        propositions = '\n'.join(lines[1:])
        if sentence and propositions:
            knowledge_base.append({'sentence': sentence, 'propositions': propositions})
    
    print(f"成功从 '{file_path}' 加载了 {len(knowledge_base)} 条知识库条目。")
    return knowledge_base

def format_knowledge_for_prompt(knowledge_base: list) -> str:
    """将加载的知识库格式化为可以注入到提示词中的字符串。"""
    if not knowledge_base:
        return ""

    formatted_examples = []
    # 在原有案例后面追加，作为补充学习材料
    header = "\n\n--- 知识库高级案例学习 (基于人工修正) ---\n"
    
    for i, item in enumerate(knowledge_base, 1):
        example = (
            f"知识库案例 {i}:\n"
            f"英文句子: \"{item['sentence']}\"\n"
            f"正确的命题条件:\n{item['propositions']}"
        )
        formatted_examples.append(example)
    
    return header + "\n\n".join(formatted_examples)



def reset_agents_to_default():
    """将全局权重和表现重置为初始默认值"""
    global agent_weights, agent_performance
    agent_weights = json.loads(json.dumps(DEFAULT_AGENT_WEIGHTS))
    agent_performance = json.loads(json.dumps(DEFAULT_AGENT_PERFORMANCE))
    print("Agent权重和表现已初始化为默认值。")

def save_weights_and_performance():
    """将当前运行的最终权重和表现保存到文件"""
    data = {'weights': agent_weights, 'performance': agent_performance}
    with open('agent_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n最终权重和表现已保存到 agent_data.json。")

def update_agent_weights():
    """根据当前运行的表现动态更新内存中的权重"""
    print("\n更新Agent权重...")
    for agent, stats in agent_performance.items():
        total = stats.get('total', 0)
        correct = stats.get('correct', 0)
        if total > 0:
            accuracy = correct / total
            base_weight = DEFAULT_AGENT_WEIGHTS.get(agent, 1.0)
            agent_weights[agent] = max(0.5, min(1.5, base_weight + (accuracy - 0.5)))
            print(f"  - {agent}: {agent_weights[agent]:.2f} (准确率: {accuracy:.2f})")
    
    calls = agent_performance['bot7'].get('calls', 0)
    if calls > 0:
        base_weight = DEFAULT_AGENT_WEIGHTS.get('bot7', 1.0)
        bonus = (calls // 10) * 0.05
        agent_weights['bot7'] = min(1.5, base_weight + bonus)
        print(f"  - bot7 (GPT-4o): {agent_weights['bot7']:.2f} (基于调用次数: {calls})")



def generate_with_gpt4o(system_instruction, user_prompt):
    """使用GPT-4o进行生成任务（后备方案）"""
    agent_performance['bot7']['calls'] += 1
    agent_performance['bot7']['total'] += 1
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_prompt}],
            temperature=0.7,
            max_tokens=500
        )
        output = response.choices[0].message.content.strip()
        print(f"GPT-4o 生成输出: {output!r}")
        return output
    except Exception as e:
        print(f"GPT-4o 生成失败: {e}")
        return ""

def generate_propositions_with_gpt4o(statement):
    return generate_with_gpt4o(system_instruction_generate, f"从以下句子中提取命题条件: {statement}")

def generate_pptl_with_gpt4o(statement, propositions):
    input_text = f"自然语言陈述: {statement}\n" + "\n".join(propositions)
    return generate_with_gpt4o(system_instruction_pptl, input_text)

def reflect_and_validate(context: str, proposal: str, task_type: str) -> (bool, str):
    """使用GPT-4o进行反思审查，根据task_type选择不同的指令"""
    print(f"\n--- [反思环节] GPT-4o 正在审查任务: {task_type.upper()} ---")
    agent_performance['bot7']['calls'] += 1
    
    if task_type == 'propositions':
        system_prompt = system_instruction_reflect_propositions
    elif task_type == 'pptl':
        system_prompt = system_instruction_reflect_pptl
    else:
        error_msg = f"内部错误：未知的审查任务类型 '{task_type}'"
        print(error_msg)
        return False, error_msg

    user_prompt = f"请严格审查以下提案。\n\n上下文:\n{context}\n\n待审查提案:\n{proposal}"
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.2,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        result_json = json.loads(content)
        is_correct = result_json.get('is_correct', False)
        feedback = result_json.get('feedback', '无具体反馈。')
        print(f"GPT-4o 反思结果: {'通过' if is_correct else '不通过'}")
        if not is_correct:
            print(f"反馈: {feedback}")
        return is_correct, feedback
    except Exception as e:
        print(f"GPT-4o 反思验证失败: {e}")
        return False, f"反思API调用失败: {e}"



@register_tool('my_file_writer')
class MyFileWriter(BaseTool):
    description = '将内容写入文件的工具。'
    parameters = [
        {'name': 'file_path', 'type': 'string', 'description': '文件路径', 'required': True},
        {'name': 'content', 'type': 'string', 'description': '要写入的内容', 'required': True}
    ]
    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json5.loads(params)
            file_path, content = os.path.basename(params_dict['file_path']), params_dict['content']
            mode = 'a' if os.path.exists(file_path) else 'w'
            with open(file_path, mode, encoding='utf-8') as f:
                f.write(content + '\n\n')
            return json5.dumps({'status': 'success', 'message': f'内容已写入 {file_path}'}, ensure_ascii=False)
        except Exception as e:
            return json5.dumps({'status': 'error', 'message': f'写入 {file_path} 失败: {str(e)}'}, ensure_ascii=False)



def normalize_output(output, is_pptl=False):
    """规范化输出，并对命题进行排序以便比较"""
    if not output: return ""
    if is_pptl:
        return output.strip().split('\n')[0].strip()
    
    output = re.sub(r'\s*\[tool_call\].*?\[tool_response\]\n.*?(?=\n|$)', '', output, flags=re.DOTALL)
    lines = [line.strip().rstrip('.') for line in output.split('\n') if line.strip() and re.match(r'^[P-Z]:', line)]
    return '\n'.join(sorted(lines))

def get_agent_response(bot, messages, bot_id):
    """获取单个Agent的响应并记录调用次数"""
    response_text = ""
    agent_performance[bot_id]['calls'] += 1
    try:
        for response in bot.run(messages=messages):
            response_text = typewriter_print(response, response_text)
    except Exception as e:
        print(f"Agent {bot_id} 调用失败: {e}")
        response_text = ""
    return response_text

# --- 辩论函数 ---

def debate_for_best_output(statement, raw_outputs, bots, max_rounds=3):
    """命题辩论"""
    vote_input = "\n".join(f"{i+1}: {output}" for i, output in enumerate(raw_outputs))
    debate_history = []

    for round_num in range(max_rounds):
        print(f"\n命题辩论 - 第 {round_num + 1} 轮")
        votes, reasons = [], []
     
        messages = [{"role": "user", "content": f"Original sentence: {statement}\nNumbered outputs:\n{vote_input}\n" + (f"Previous round votes:\n{debate_history[-1]}" if debate_history else "") + "\nSelect the best number (1, 2, or 3) and provide your reason."}]
        
      
        original_prompts = [bot.system_message for bot in bots]
        for bot in bots:
         
            bot.system_message = system_instruction_debate

        for i, bot in enumerate(bots):
            bot_id = f'bot{i+1}'
            response_text = get_agent_response(bot, messages, bot_id)
            match = re.match(r'Vote:\s*(\d+)\s*Reason:\s*(.+)', response_text, re.DOTALL)
            
            if match and match.group(1) in ['1', '2', '3']:
                vote, reason = int(match.group(1)), match.group(2).strip()
                votes.append(vote)
                reasons.append(f"Agent {i+1} 投了 {vote}: 理由: {reason}")
                print(f"Agent {i+1} 投了 {vote}: 理由: {reason}")
                agent_performance[bot_id]['total'] += 1
                if round_num == max_rounds - 1 or (len(votes) == 3 and all(v == votes[0] for v in votes)):
                    agent_performance[bot_id]['correct'] += 1
            else:
                votes.append(None)
                reasons.append(f"Agent {i+1} 投票无效")
                print(f"Agent {i+1} 投票无效: {response_text!r}")

        # 恢复原始系统指令
        for i, bot in enumerate(bots):
            bot.system_message = original_prompts[i]
        
        debate_history.append("\n".join(reasons))
        
        if all(v == votes[0] and v is not None for v in votes) and votes:
            print(f"在第 {round_num + 1} 轮达成共识: 投票 {votes[0]}")
            return votes[0] - 1

    print("辩论未达成共识")
    return None

def debate_for_best_pptl(statement, propositions, pptl_outputs, bots, max_rounds=3):
    """PPTL公式辩论"""
    vote_input = "\n".join(f"{i+1}: {output}" for i, output in enumerate(pptl_outputs))
    propositions_str = "\n".join(propositions)
    debate_history = []

    for round_num in range(max_rounds):
        print(f"\nPPTL辩论 - 第 {round_num + 1} 轮")
        votes, reasons = [], []
        messages = [{"role": "user", "content": f"Natural statement: {statement}\nProposition conditions:\n{propositions_str}\nNumbered outputs:\n{vote_input}\n" + (f"Previous round votes:\n{debate_history[-1]}" if debate_history else "") + "\nSelect the best number (1, 2, or 3) and provide your reason."}]
        
        original_prompts = [bot.system_message for bot in bots]
        for bot in bots:
            bot.system_message = system_instruction_pptl_debate
        
        for i, bot in enumerate(bots):
            bot_id = f'bot{i+4}'
            response_text = get_agent_response(bot, messages, bot_id)
            match = re.match(r'Vote:\s*(\d+)\s*Reason:\s*(.+)', response_text, re.DOTALL)
            
            if match and match.group(1) in ['1', '2', '3']:
                vote, reason = int(match.group(1)), match.group(2).strip()
                votes.append(vote)
                reasons.append(f"Agent {i+4} 投了 {vote}: 理由: {reason}")
                print(f"Agent {i+4} 投了 {vote}: 理由: {reason}")
                agent_performance[bot_id]['total'] += 1
                if round_num == max_rounds - 1 or (len(votes) == 3 and all(v == votes[0] for v in votes)):
                    agent_performance[bot_id]['correct'] += 1
            else:
                votes.append(None)
                reasons.append(f"Agent {i+4} 投票无效")
                print(f"Agent {i+4} 投票无效: {response_text!r}")

        for i, bot in enumerate(bots):
            bot.system_message = original_prompts[i]

        debate_history.append("\n".join(reasons))

        if all(v == votes[0] and v is not None for v in votes) and votes:
            print(f"在第 {round_num + 1} 轮达成共识: 投票 {votes[0]}")
            return votes[0] - 1

    print("辩论未达成共识")
    return None



def process_statement(statement, bots, max_attempts=3):
    """处理单个语句，提取命题，包含共识强度判断和带反馈的修正循环"""
    file_writer = MyFileWriter()
    
    feedback_from_last_round = ""

    for attempt in range(1, max_attempts + 1):
        print(f"\n{'='*20}\n处理语句: {statement}\n尝试次数: {attempt}/{max_attempts}\n{'='*20}")
        
        base_prompt = f"从以下句子中提取命题条件: {statement}"
        if feedback_from_last_round:
            prompt_with_feedback = (
                f"{base_prompt}\n\n"
                f"**重要修正指南**: 上一轮的方案被审查员指出存在以下问题，请在本次生成中务必修正：\n"
                f"“{feedback_from_last_round}”\n"
                f"请根据此指南生成一个更正后的版本。"
            )
            messages = [{'role': 'user', 'content': prompt_with_feedback}]
            print(f"--- 正在使用上一轮的反馈进行重试 ---\n")
        else:
            messages = [{'role': 'user', 'content': base_prompt}]
        
        raw_outputs = [get_agent_response(bot, messages, f'bot{i+1}') for i, bot in enumerate(bots)]
        
        normalized_outputs = [normalize_output(out) for out in raw_outputs]
        valid_outputs = [out for out in normalized_outputs if out]
        
        if not valid_outputs:
            print("本轮所有Agent均未生成有效输出，重试..."); time.sleep(2); continue

        output_counts = Counter(valid_outputs)
        most_common = output_counts.most_common(1)[0]
        num_valid = len(valid_outputs)
        most_common_count = most_common[1]
        
        final_output_normalized = None
        needs_reflection = True

        if most_common_count == 3 or (most_common_count == 2 and num_valid == 2):
            print(f"\n--> 达成强共识 ({most_common_count}/{num_valid} 一致)，豁免审查，直接采纳。")
            final_output_normalized = most_common[0]
            needs_reflection = False
        elif most_common_count == 2:
            print(f"\n--> 达成中等共识 (2/{num_valid} 一致)，采纳多数派方案并提交审查。")
            final_output_normalized = most_common[0]
        else:
            print("\n--> 未达成共识，进入完整辩论流程。")
            best_index = debate_for_best_output(statement, raw_outputs, bots)
            if best_index is not None:
                print(f"辩论选出 Agent {best_index+1} 的方案，提交审查。")
                final_output_normalized = normalized_outputs[best_index]
            else:
                print("辩论失败，将进行下一轮尝试。"); time.sleep(2); continue
        
        if final_output_normalized is not None:
            if needs_reflection:
                # 找到一个未规范化的原始输出用于审查
                original_format_output = raw_outputs[normalized_outputs.index(final_output_normalized)]
                context = f"Original sentence: {statement}"
                is_correct, feedback = reflect_and_validate(context, original_format_output, task_type='propositions')
                if not is_correct:
                    feedback_from_last_round = feedback 
                    print(f"方案未通过审查，将携带反馈进入下一轮尝试。"); time.sleep(2); continue
            
            content = f"{statement}\n{final_output_normalized}"
            result = file_writer.call(json5.dumps({'file_path': 'prop.txt', 'content': content}))
            print(f"\n*** 成功保存命题: {json5.loads(result)['message']} ***")
            return True

    print(f"\n所有 {max_attempts} 次尝试失败，调用GPT-4o作为最终手段...")
    gpt_output_raw = generate_propositions_with_gpt4o(statement)
    gpt_output_normalized = normalize_output(gpt_output_raw)
    if gpt_output_normalized:
        context = f"Original sentence: {statement}"
        is_correct, _ = reflect_and_validate(context, gpt_output_raw, task_type='propositions')
        if is_correct:
             content = f"{statement}\n{gpt_output_normalized}"
             result = file_writer.call(json5.dumps({'file_path': 'prop.txt', 'content': content}))
             print(f"\n*** 成功保存GPT-4o命题: {json5.loads(result)['message']} ***")
             return True
    
    print(f"\n*** 最终失败：未能为语句 '{statement}' 生成有效命题。***")
    with open('failure_prop.txt', 'a', encoding='utf-8') as f_fail:
        f_fail.write(statement + '\n')
    return False

def convert_to_pptl(bots, max_attempts=3):
    """读取命题，转换为PPTL公式，包含共识强度判断和反馈循环"""
    try:
        with open('prop.txt', 'r', encoding='utf-8') as f: content = f.read()
    except FileNotFoundError:
        print("prop.txt文件未找到，无法转换。"); return False

    entries = [entry.strip() for entry in content.split('\n\n') if entry.strip()]
    if not entries:
        print("prop.txt文件为空，无法转换。"); return False

    pptl_results = []
    file_writer = MyFileWriter()
    
    for entry in entries:
        lines = entry.split('\n')
        statement, propositions = lines[0], lines[1:]
        
        feedback_from_last_round = ""
        success = False
        
        for attempt in range(1, max_attempts + 1):
            print(f"\n{'='*20}\n转换PPTL: {statement}\n尝试次数: {attempt}/{max_attempts}\n{'='*20}")
            
            base_input_text = f"自然语言陈述: {statement}\n" + "\n".join(propositions)
            if feedback_from_last_round:
                prompt_with_feedback = (
                    f"{base_input_text}\n\n"
                    f"**重要修正指南**: 上一轮的PPTL公式被审查员指出存在以下问题，请在本次生成中务必修正：\n"
                    f"“{feedback_from_last_round}”\n"
                    f"请根据此指南生成一个更正后的PPTL公式。"
                )
                messages = [{'role': 'user', 'content': prompt_with_feedback}]
                print(f"--- 正在使用上一轮的反馈进行重试 ---\n")
            else:
                messages = [{'role': 'user', 'content': base_input_text}]
            
            pptl_outputs = [get_agent_response(bot, messages, f'bot{i+4}') for i, bot in enumerate(bots)]
            
            normalized_pptl_outputs = [normalize_output(out, is_pptl=True) for out in pptl_outputs]
            valid_pptl_outputs = [out for out in normalized_pptl_outputs if out]

            if not valid_pptl_outputs:
                print("本轮所有Agent均未生成有效PPTL，重试..."); time.sleep(2); continue
            
            output_counts = Counter(valid_pptl_outputs)
            most_common = output_counts.most_common(1)[0]
            num_valid = len(valid_pptl_outputs)
            most_common_count = most_common[1]

            final_pptl = None
            needs_reflection = True

            if most_common_count == 3 or (most_common_count == 2 and num_valid == 2):
                print(f"\n--> 达成PPTL强共识 ({most_common_count}/{num_valid} 一致)，豁免审查，直接采纳。")
                final_pptl = most_common[0]
                needs_reflection = False
            elif most_common_count == 2:
                print(f"\n--> 达成PPTL中等共识 (2/{num_valid} 一致)，采纳多数派方案并提交审查。")
                final_pptl = most_common[0]
            else:
                print("\n--> 未达成PPTL共识，进入完整辩论流程。")
                best_index = debate_for_best_pptl(statement, propositions, normalized_pptl_outputs, bots)
                if best_index is not None:
                    print(f"辩论选出 Agent {best_index+4} 的方案，提交审查。")
                    final_pptl = normalized_pptl_outputs[best_index]
                else:
                    print("辩论失败，将进行下一轮尝试。"); time.sleep(2); continue
            
            if final_pptl is not None:
                if needs_reflection:
                    context = f"自然语言陈述: {statement}\n命题:\n" + "\n".join(propositions)
                    is_correct, feedback = reflect_and_validate(context, final_pptl, task_type='pptl')
                    if not is_correct:
                        feedback_from_last_round = feedback
                        print(f"PPTL方案未通过审查，将携带反馈进入下一轮尝试。"); time.sleep(2); continue
                
                pptl_results.append(f"{statement}\nPPTL: {final_pptl}")
                print(f"\n*** 成功获得PPTL: {final_pptl} ***")
                success = True
                break

        if not success:
            print(f"\n所有 {max_attempts} 次尝试失败，为 '{statement}' 调用GPT-4o后备...")
            gpt_pptl_raw = generate_pptl_with_gpt4o(statement, propositions)
            gpt_pptl_normalized = normalize_output(gpt_pptl_raw, is_pptl=True)
            if gpt_pptl_normalized:
                context = f"自然语言陈述: {statement}\n命题:\n" + "\n".join(propositions)
                is_correct, _ = reflect_and_validate(context, gpt_pptl_normalized, task_type='pptl')
                if is_correct:
                    pptl_results.append(f"{statement}\nPPTL: {gpt_pptl_normalized}")
                    print(f"\n*** 成功获得GPT-4o PPTL: {gpt_pptl_normalized} ***")
                else:
                    print(f"\n*** 最终失败：GPT-4o的PPTL也未通过审查，跳过 '{statement}'。***")
                    with open('failure_pptl.txt', 'a', encoding='utf-8') as f_fail:
                        f_fail.write(statement + '\n')
            else:
                print(f"\n*** 最终失败：GPT-4o未能生成有效PPTL，跳过 '{statement}'。***")
                with open('failure_pptl.txt', 'a', encoding='utf-8') as f_fail:
                    f_fail.write(statement + '\n')

    if pptl_results:
        content_to_save = "\n\n".join(pptl_results)
        if os.path.exists('pptl.txt'): os.remove('pptl.txt')
        result = file_writer.call(json5.dumps({'file_path': 'pptl.txt', 'content': content_to_save}))
        print(f"\n所有PPTL公式已生成: {json5.loads(result)['message']}")
        return True
    else:
        print("\n没有生成任何PPTL公式。")
        return False


if __name__ == "__main__":
   
    reset_agents_to_default()
    
    

    
    KNOWLEDGE_FILE = 'knowledges_prop.txt'
    knowledge_base = load_knowledge_base(KNOWLEDGE_FILE)
    knowledge_prompt_section = format_knowledge_for_prompt(knowledge_base)
    
   
    if knowledge_prompt_section:
        print("正在使用知识库内容增强相关系统指令...")
        
        system_instruction_generate += knowledge_prompt_section
        system_instruction_reflect_propositions += knowledge_prompt_section
        system_instruction_debate += knowledge_prompt_section
        print("指令增强完成。")
    


  
    DASHSCOPE_API_KEY = "" 
    DEEPSEEK_API_KEY = "" 
   
    llm_cfg_1 = {'model': 'qwen-turbo', 'model_server': 'dashscope', 'api_key': DASHSCOPE_API_KEY, 'generate_cfg': {'top_p': 0.8, 'seed': random.randint(1, 10000)}}
    llm_cfg_2 = {
        'model': 'deepseek-chat', 
        'model_server': 'https://api.deepseek.com/v1', 
        'api_key': DEEPSEEK_API_KEY, 
        'generate_cfg': {'top_p': 0.85, 'seed': random.randint(1, 10000)}
    }
    llm_cfg_3 = {'model': 'qwen-max', 'model_server': 'dashscope', 'api_key': DASHSCOPE_API_KEY, 'generate_cfg': {'top_p': 0.9, 'seed': random.randint(1, 10000)}}
    llm_cfg_4 = {'model': 'qwen-turbo', 'model_server': 'dashscope', 'api_key': DASHSCOPE_API_KEY, 'generate_cfg': {'top_p': 0.8, 'seed': random.randint(1, 10000)}}
    llm_cfg_5 = {
        'model': 'deepseek-chat', 
        'model_server': 'https://api.deepseek.com/v1', 
        'api_key': DEEPSEEK_API_KEY, 
        'generate_cfg': {'top_p': 0.85, 'seed': random.randint(1, 10000)}
    }
    llm_cfg_6 = {'model': 'qwen-max', 'model_server': 'dashscope', 'api_key': DASHSCOPE_API_KEY, 'generate_cfg': {'top_p': 0.9, 'seed': random.randint(1, 10000)}}

    tools = ['my_file_writer']
    
   
    prop_bots = [
        Assistant(llm=llm_cfg_1, system_message=system_instruction_generate, function_list=tools),
        Assistant(llm=llm_cfg_2, system_message=system_instruction_generate, function_list=tools),
        Assistant(llm=llm_cfg_3, system_message=system_instruction_generate, function_list=tools)
    ]
    
    
    pptl_bots = [
        Assistant(llm=llm_cfg_4, system_message=system_instruction_pptl, function_list=tools),
        Assistant(llm=llm_cfg_5, system_message=system_instruction_pptl, function_list=tools),
        Assistant(llm=llm_cfg_6, system_message=system_instruction_pptl, function_list=tools)
    ]
    

    input_file = 'input.txt'
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            statements = [line.strip() for line in f if line.strip()]
        if not statements:
            print(f"{input_file} 为空，退出。"); exit(0)
    except FileNotFoundError:
        print(f"{input_file} 未找到，退出。"); exit(1)

    for file in ['prop.txt', 'pptl.txt','failure_prop.txt', 'failure_pptl.txt']:
        if os.path.exists(file):
            os.remove(file)
            print(f"已清理旧文件 {file}")

    print(f"从 {input_file} 读取了 {len(statements)} 条语句。")
    

    for stmt in statements:
        process_statement(stmt, prop_bots)
        update_agent_weights()

    # 阶段二：转换为PPTL
    convert_to_pptl(pptl_bots)
    update_agent_weights()

    print("\n所有任务完成。")
    

    save_weights_and_performance()
