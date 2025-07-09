import abc
import asyncio
import time
from typing import Any

import yaml
from litellm import acompletion
from loguru import logger
from pydantic import BaseModel

from ardcore.data.types import Triplet


# TODO: use this approach in `SwarmConfig`, then remove those 3 classes
class SwarmKGConfig(abc.ABC, BaseModel):
    scientific_domain: str = "bioscience"  # Default value
    prompt_repository: str
    merging_model_name: str | None = None
    merging_model_params: dict[str, Any] = {}


class HomogeneousSwarmKGConfig(SwarmKGConfig):
    extractor_model_name: str
    extractor_model_params: dict[str, Any] = {}
    swarm_size: int = 1


class HeterogeneousSwarmKGConfig(SwarmKGConfig):
    extractor_model_names: list[str]
    extractor_model_params: dict[str, Any] = {}


class TripletList(BaseModel):
    triplets: list[Triplet]


def extract_swarm_generator(context, config: dict) -> list[Triplet]:
    local_config = config.copy()
    merging_model_name = local_config.pop("merging_model_name", None)
    merging_model_params = local_config.pop("merging_model_params", {})

    if "extractor_model_name" in local_config:
        extractor_model_name = local_config.pop("extractor_model_name")
        extractor_model_params = local_config.pop("extractor_model_params", {})
        swarm_size = local_config.pop("swarm_size", 1)
        pipeline_config = HomogeneousSwarmKGConfig(
            extractor_model_name=extractor_model_name,
            extractor_model_params=extractor_model_params,
            swarm_size=swarm_size,
            merging_model_name=merging_model_name,
            merging_model_params=merging_model_params,
            **local_config,
        )
    elif "extractor_model_names" in local_config:
        extractor_model_names = local_config.pop("extractor_model_names")
        extractor_model_params = local_config.pop("extractor_model_params", {})
        pipeline_config = HeterogeneousSwarmKGConfig(
            extractor_model_names=extractor_model_names,
            extractor_model_params=extractor_model_params,
            merging_model_name=merging_model_name,
            merging_model_params=merging_model_params,
            **local_config,
        )
    else:
        raise ValueError(
            "Configuration must contain either 'extractor_model_name' (for homogeneous) "
            "or 'extractor_model_names' (for heterogeneous)"
        )

    pipeline = SwarmKnowledgeGraphPipeline.from_config(pipeline_config)

    start_time = time.time()
    result = asyncio.run(pipeline.run(context))
    logger.debug(
        f"Knowledge graph generation completed in {time.time() - start_time:.2f} seconds"
    )
    return result.get("merged_result", [])


class SwarmKnowledgeGraphPipeline:
    def __init__(
        self,
        scientific_domain: str,
        prompt_repository: str,
        extractor_model_name: str | None = None,
        extractor_model_names: list[str] | None = None,
        extractor_model_params: dict[str, Any] = {},
        merging_model_name: str | None = None,
        merging_model_params: dict[str, Any] = {},
        swarm_size: int = 1,
    ):
        self.scientific_domain = scientific_domain
        self.prompt_repository = PromptRepository(prompt_repository)
        self.extractor_model_name = extractor_model_name
        self.extractor_model_names = extractor_model_names or []
        self.extractor_model_params = extractor_model_params
        self.merging_model_name = merging_model_name
        self.merging_model_params = merging_model_params
        self.swarm_size = swarm_size

        if not self.extractor_model_name and not self.extractor_model_names:
            raise ValueError(
                "Either extractor_model_name or extractor_model_names must be provided"
            )
        if self.extractor_model_name and self.extractor_model_names:
            raise ValueError(
                "Provide either extractor_model_name or extractor_model_names, not both"
            )

    @classmethod
    def from_config(
        cls,
        config: SwarmKGConfig,
    ) -> "SwarmKnowledgeGraphPipeline":
        """
        Create an instance of SwarmKnowledgeGraphPipeline from a configuration Pydantic model.
        Store the original config on the instance.
        """
        config_raw = config.model_copy(deep=True)

        if isinstance(config, HomogeneousSwarmKGConfig):
            instance = cls(
                scientific_domain=config.scientific_domain,
                prompt_repository=config.prompt_repository,
                extractor_model_name=config.extractor_model_name,
                extractor_model_params=config.extractor_model_params,
                swarm_size=config.swarm_size,
                merging_model_name=config.merging_model_name,
                merging_model_params=config.merging_model_params,
            )
        elif isinstance(config, HeterogeneousSwarmKGConfig):
            instance = cls(
                scientific_domain=config.scientific_domain,
                prompt_repository=config.prompt_repository,
                extractor_model_names=config.extractor_model_names,
                extractor_model_params=config.extractor_model_params,
                merging_model_name=config.merging_model_name,
                merging_model_params=config.merging_model_params,
            )
        else:
            raise TypeError("Unsupported configuration type passed to from_config")

        instance.config = config_raw
        return instance

    async def extract_triplets(
        self,
        markdown_text: str,
    ) -> list[str]:
        """
        Execute extraction using litellm.acompletion based on the configured models.
        """
        sys_msg = self.prompt_repository.get_prompt("swarm_kg_system_message")
        prompt = self.prompt_repository.get_prompt(
            "swarm_kg_prompt", paper_content=markdown_text
        )
        messages = [
            {"content": sys_msg, "role": "system"},
            {"content": prompt, "role": "user"},
        ]

        logger.info("Starting extraction of knowledge triplets using litellm")
        tasks = []

        if self.extractor_model_name:
            model_names = [self.extractor_model_name] * self.swarm_size
            logger.info(
                f"Using homogeneous swarm: {self.extractor_model_name} x {self.swarm_size}"
            )
        else:
            model_names = self.extractor_model_names
            logger.info(f"Using heterogeneous swarm: {model_names}")

        for model_name in model_names:
            logger.debug(
                f"Creating task for model: {model_name} with params {self.extractor_model_params}"
            )
            tasks.append(
                acompletion(
                    model=model_name, messages=messages, **self.extractor_model_params
                )
            )

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        extracted_contents = []
        for i, response in enumerate(responses, start=1):
            if isinstance(response, Exception):
                logger.error(
                    f"Error in litellm call {i} ({model_names[i - 1]}): {response}"
                )
                extracted_contents.append(f"Error processing response {i}")
            elif (
                response
                and response.choices
                and response.choices[0].message
                and response.choices[0].message.content
            ):
                content = response.choices[0].message.content
                logger.debug(f"Response {i} ({model_names[i - 1]}): {content[:100]}...")
                extracted_contents.append(content)
            else:
                logger.warning(
                    f"Received empty or invalid response {i} ({model_names[i - 1]}): {response}"
                )
                extracted_contents.append(f"Empty response {i}")

        return extracted_contents

    async def merge_triplets(
        self,
        triplet_json_list: list[str],
    ) -> list[Triplet]:
        triplets_str = "\n".join(triplet_json_list)
        prompt = self.prompt_repository.get_prompt(
            "prompt_merge_llm", triplets=triplets_str
        )

        sys_msg = self.prompt_repository.get_prompt(
            "sys_msg_generic", domain=self.scientific_domain
        )

        messages = [
            {"content": sys_msg, "role": "system"},
            {"content": prompt, "role": "user"},
        ]

        logger.info(f"Merging triplets using litellm model: {self.merging_model_name}")

        try:
            # LLM Call with structured output
            response = await acompletion(
                model=self.merging_model_name,
                messages=messages,
                response_format=TripletList,  # Pass the Pydantic model directly
                **self.merging_model_params,  # Pass additional params
            )

            # Extract the parsed Pydantic object
            # When response_format is a Pydantic model, litellm returns the JSON string
            # in the content field. We need to parse it manually.
            if (
                response
                and response.choices
                and response.choices[0].message
                and isinstance(response.choices[0].message.content, str)
            ):
                json_string = response.choices[0].message.content
                logger.debug(
                    f"Received string content for merge: {json_string[:100]}..."
                )
                try:
                    # Attempt to parse the string content into the Pydantic model
                    result = TripletList.model_validate_json(json_string)
                    logger.debug(f"Successfully parsed merged result: {result}")
                    return result.triplets
                except Exception as parse_error:
                    logger.error(
                        f"Manual JSON parsing failed: {parse_error}. Content: {json_string}"
                    )
                    return []  # Return empty list on parsing failure
            else:
                # Log unexpected response structure (e.g., content is not a string or response is malformed)
                logger.error(
                    f"Failed to get valid string content from litellm response for merge. Response: {response}"
                )
                return []  # Return empty list if no valid string content

        except Exception as e:
            logger.exception(f"Error during litellm merge call: {e}")
            return []

    async def run(
        self,
        markdown_text: str,
    ) -> dict:
        pipeline_result = {}

        extraction_results = await self.extract_triplets(markdown_text)
        pipeline_result["extraction_results"] = extraction_results

        merged_result_triplets = []
        if self.merging_model_name:
            merged_result_triplets = await self.merge_triplets(extraction_results)
            pipeline_result["merged_result"] = merged_result_triplets
        else:
            logger.info("No merging model specified, skipping merge step.")
            pipeline_result["merged_result"] = []

        logger.info(
            f"Pipeline finished. Extracted {len(extraction_results)} sets, Merged into {len(merged_result_triplets)} triplets."
        )
        return pipeline_result


class PromptRepository:
    def __init__(self, prompt_file: str):
        with open(prompt_file, "r") as file:
            self.prompts = yaml.safe_load(file)

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        template = self.prompts.get(prompt_name)
        if template is None:
            raise ValueError(f"Prompt {prompt_name} not found in repository")
        return template.format(**kwargs)
