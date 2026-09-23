TPO_LOSS_SYSTEM_TEMPLATE = """You are a language model tasked with evaluating a chosen response by comparing it with a rejected response to a user query. Analyze the strengths and weaknesses of each response, step by step, and explain why one is chosen or rejected.

**User Query**:
{query}

**Rejected Response**:
{rejected_response}

**Do NOT generate a response to the query. Be concise.** Below is the chosen response."""


TEXTGRAD_BACKWARD_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves a given text (i.e. the variable). You are the gradient "
    "(feedback) engine. Your only responsibility is to give intelligent and creative feedback and constructive "
    "criticism to variables, given an objective specified in <OBJECTIVE_FUNCTION> </OBJECTIVE_FUNCTION> tags. "
    "The variables may be solutions to problems, prompts to language models, code, or any other text-based variable. "
    "Pay attention to the role description of the variable, and the context in which it is used. You should assume "
    "that the variable will be used in a similar context in the future. Only provide strategies, explanations, and "
    "methods to change in the variable. DO NOT propose a new version of the variable, that will be the job of the "
    "optimizer. Your only job is to send feedback and criticism (compute 'gradients'). If a variable is already "
    "working well, you should not give feedback."
)


TEXTGRAD_OPTIMIZER_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., variable). You will be asked to creatively "
    "and critically improve prompts, solutions to problems, code, or any other text-based variable. You will receive "
    "some feedback, and use the feedback to improve the variable. The feedback may be noisy; identify what is "
    "important and correct. Pay attention to the role description of the variable and the context in which it is "
    "used. You MUST give your response by sending the improved variable between <IMPROVED_VARIABLE> and "
    "</IMPROVED_VARIABLE> tags. The text between the tags will directly replace the variable."
)


def build_tpo_loss_system_prompt(query: str, rejected_response: str) -> str:
    return TPO_LOSS_SYSTEM_TEMPLATE.format(
        query=query.strip(), rejected_response=rejected_response.strip()
    )


def build_tpo_backward_prompt(
    loss_system_prompt: str, chosen_response: str, textual_loss: str
) -> str:
    conversation = (
        f"<LM_SYSTEM_PROMPT> {loss_system_prompt} </LM_SYSTEM_PROMPT>\n\n"
        f"<LM_INPUT> {chosen_response.strip()} </LM_INPUT>\n\n"
        f"<LM_OUTPUT> {textual_loss.strip()} </LM_OUTPUT>\n\n"
    )
    return (
        "You will give feedback to a variable with the following role: "
        "<ROLE> a chosen response to a user query </ROLE>. "
        "Here is an evaluation of the variable using a language model:\n\n"
        f"{conversation}"
        "<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the variable given the above evaluation "
        "output. Our only goal is to improve the above metric, and nothing else. </OBJECTIVE_FUNCTION>\n\n"
        "We are interested in giving feedback to the a chosen response to a user query for this conversation. "
        "Specifically, give feedback to the following span of text:\n\n"
        f"<VARIABLE> {chosen_response.strip()} </VARIABLE>\n\n"
        "Given the above history, describe how the a chosen response to a user query could be improved to improve "
        "the <OBJECTIVE_FUNCTION>. Be very creative, critical, and intelligent."
    )


def build_tpo_update_prompt(
    loss_system_prompt: str,
    chosen_response: str,
    textual_loss: str,
    textual_gradient: str,
) -> str:
    conversation = (
        f"<LM_SYSTEM_PROMPT> {loss_system_prompt} </LM_SYSTEM_PROMPT>\n\n"
        f"<LM_INPUT> {chosen_response.strip()} </LM_INPUT>\n\n"
        f"<LM_OUTPUT> {textual_loss.strip()} </LM_OUTPUT>\n\n"
    )
    gradient_context = (
        f"Here is a conversation:\n\n<CONVERSATION>{conversation}</CONVERSATION>\n\n"
        "This conversation is potentially part of a larger system. The output is used as response from the language "
        "model\n\nHere is the feedback we got for a chosen response to a user query in the conversation:\n\n"
        f"<FEEDBACK>{textual_gradient.strip()}</FEEDBACK>\n\n"
    )
    return (
        "Here is the role of the variable you will improve: "
        "<ROLE>a chosen response to a user query</ROLE>.\n\n"
        f"The variable is the text within the following span: <VARIABLE> {chosen_response.strip()} </VARIABLE>\n\n"
        "Here is the context and feedback we got for the variable:\n\n"
        f"<CONTEXT>{gradient_context}</CONTEXT>\n\n"
        "Improve the variable (a chosen response to a user query) using the feedback provided in <FEEDBACK> tags.\n"
        "You must follow the following constraints:\n\n"
        "<CONSTRAINTS>Constraint 1: Only generate a chosen response.\n"
        "Constraint 2: Do NOT generate a rejected response.</CONSTRAINTS>\n\n"
        "Send the improved variable in the following format:\n\n"
        "<IMPROVED_VARIABLE>{the improved variable}</IMPROVED_VARIABLE>\n\n"
        "Send ONLY the improved variable between the <IMPROVED_VARIABLE> tags, and nothing else."
    )
