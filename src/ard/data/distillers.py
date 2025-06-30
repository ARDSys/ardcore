import typing
from dataclasses import dataclass

from litellm import completion
from loguru import logger


class DistillerProtocol(typing.Protocol):
    def distill(self, text: str) -> str:
        pass


@dataclass
class DefaultDistiller(DistillerProtocol):
    model_name: str

    def distill(self, text: str) -> str:
        sys_msg1 = (
            "You respond with a concise scientific summary, including reasoning. "
            "You never use names or references. "
        )
        usr_prompt1 = (
            "In a matter-of-fact voice, rewrite this '{text}'. "
            "The writing must stand on its own and provide all background needed, "
            "and include details. Do not include names, figures, plots or "
            "citations in your response, only facts."
        )

        usr_prompt2 = (
            "Provide a bullet point list of the key facts and reasoning in {summary}. "
            "The writing must stand on its own and provide all background needed, and "
            "include details. Do not include figures, plots or citations in your response. Think step by step. "
        )

        try:
            summary = generate_response(
                self.model_name, sys_msg1, usr_prompt1.format(text=text)
            )
            logger.debug("Generated summary")
            bullet_list = generate_response(
                self.model_name, sys_msg1, usr_prompt2.format(summary=summary)
            )
            logger.debug("Generated bullet list")
            return summary + "\n\n" + bullet_list
        except Exception as e:
            logger.error(f"Error in text distillation: {e}")
            raise


# TODO change to using an llm
def generate_response(model_name, sys_message, user_prompt):
    messages = [
        {"content": sys_message, "role": "system"},
        {"content": user_prompt, "role": "user"},
    ]
    response = completion(model=model_name, messages=messages)
    return response.choices[0].message.content
