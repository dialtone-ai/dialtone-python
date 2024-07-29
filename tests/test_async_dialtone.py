import json
import os
from typing import AsyncGenerator
import pytest
from dialtone import AsyncDialtone
from dialtone.types import (
    LLM,
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    ProviderConfig,
    RouterModelConfig,
    Provider,
)

api_keys = {
    "dialtone": os.getenv("DIALTONE_API_KEY") or "",
    "openai": os.getenv("OPENAI_API_KEY") or "",
    "anthropic": os.getenv("ANTHROPIC_API_KEY") or "",
    "google": os.getenv("GOOGLE_API_KEY") or "",
    "groq": os.getenv("GROQ_API_KEY") or "",
    "cohere": os.getenv("COHERE_API_KEY") or "",
    "fireworks": os.getenv("FIREWORKS_API_KEY") or "",
    "deepinfra": os.getenv("DEEPINFRA_API_KEY") or "",
    "together": os.getenv("TOGETHER_API_KEY") or "",
    "replicate": os.getenv("REPLICATE_API_KEY") or "",
}


@pytest.mark.asyncio
async def test_chat_completion():
    global api_keys

    """
    Test Chat Completion API
    """
    dial_configs = [
        {"quality": 1, "cost": 0},
        {"quality": 0.75, "cost": 0.25},
        {"quality": 0.5, "cost": 0.5},
        {"quality": 0.25, "cost": 0.75},
        {"quality": 0, "cost": 1},
    ]

    print()
    print("Testing Chat Completion")
    print("-----------------------")
    for dials in dial_configs:
        dialtone = AsyncDialtone(
            api_key=api_keys["dialtone"],
            provider_config=ProviderConfig(
                openai=ProviderConfig.OpenAI(api_key=api_keys["openai"]),
                anthropic=ProviderConfig.Anthropic(api_key=api_keys["anthropic"]),
                google=ProviderConfig.Google(api_key=api_keys["google"]),
                groq=ProviderConfig.Groq(api_key=api_keys["groq"]),
                cohere=ProviderConfig.Cohere(api_key=api_keys["cohere"]),
                fireworks=ProviderConfig.Fireworks(api_key=api_keys["fireworks"]),
                deepinfra=ProviderConfig.DeepInfra(api_key=api_keys["deepinfra"]),
                together=ProviderConfig.Together(api_key=api_keys["together"]),
                replicate=ProviderConfig.Replicate(api_key=api_keys["replicate"]),
            ),
            router_model_config=RouterModelConfig(
                llama_3_70b=RouterModelConfig.Llama_3_70B(
                    no_tools_providers=[Provider.Fireworks, Provider.Groq],
                )
            ),
            dials=dials,
            base_url="http://localhost:8000",
        )

        response = await dialtone.chat.completions.create(
            messages=[ChatMessage(role="user", content="Hello, world!")]
        )

        assert isinstance(response, ChatCompletion)

        choice = response.choices[0]
        assert choice.message.content is not None
        assert choice.message.role == "assistant"
        assert response.model is not None
        assert response.provider is not None

        usage = response.usage
        assert usage.prompt_tokens is not None
        assert usage.completion_tokens is not None
        assert usage.total_tokens is not None

        print(
            f"Dials: {dials} | Model: {response.model} | Provider: {response.provider}"
        )


@pytest.mark.asyncio
async def test_chat_completion_with_tools():
    global api_keys

    """
    Test Chat Completion API with tools
    """
    dial_configs = [
        {"quality": 0.5, "cost": 0.5},
    ]

    print()
    print("Testing Chat Completion with Tools")
    print("---------------------------------")
    for dials in dial_configs:
        dialtone = AsyncDialtone(
            api_key=api_keys["dialtone"],
            provider_config=ProviderConfig(
                openai=ProviderConfig.OpenAI(api_key=api_keys["openai"]),
                anthropic=ProviderConfig.Anthropic(api_key=api_keys["anthropic"]),
                google=ProviderConfig.Google(api_key=api_keys["google"]),
                groq=ProviderConfig.Groq(api_key=api_keys["groq"]),
                cohere=ProviderConfig.Cohere(api_key=api_keys["cohere"]),
                fireworks=ProviderConfig.Fireworks(api_key=api_keys["fireworks"]),
                deepinfra=ProviderConfig.DeepInfra(api_key=api_keys["deepinfra"]),
                together=ProviderConfig.Together(api_key=api_keys["together"]),
                replicate=ProviderConfig.Replicate(api_key=api_keys["replicate"]),
            ),
            router_model_config=RouterModelConfig(
                llama_3_70b=RouterModelConfig.Llama_3_70B(
                    no_tools_providers=[Provider.Fireworks, Provider.Groq],
                )
            ),
            dials=dials,
            base_url="http://localhost:8000",
        )

        messages = [
            {
                "role": "user",
                "content": "What's the weather like in San Francisco, Tokyo, and Paris?",
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = await dialtone.chat.completions.create(
            messages=messages,
            tools=tools,
        )

        assert isinstance(response, ChatCompletion)

        choice = response.choices[0]
        assert choice.message.content is not None
        assert choice.message.role == "assistant"
        assert response.model is not None
        assert response.provider is not None

        usage = response.usage
        assert usage.prompt_tokens is not None
        assert usage.completion_tokens is not None
        assert usage.total_tokens is not None

        print("first response", response)

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        assert tool_calls is not None and len(tool_calls) > 0

        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        messages.append(
            response_message.__dict__
        )  # extend conversation with assistant's reply

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = await dialtone.chat.completions.create(
            messages=messages,
        )  # get a new response from the model where it can see the function response

        print("second response", second_response)

        assert isinstance(second_response, ChatCompletion)

        choice = second_response.choices[0]
        assert choice.message.content is not None
        assert choice.message.role == "assistant"
        assert second_response.model is not None
        assert second_response.provider is not None

        usage = second_response.usage
        assert usage.prompt_tokens is not None
        assert usage.completion_tokens is not None
        assert usage.total_tokens is not None


@pytest.mark.asyncio
async def test_chat_completion_streaming():
    global api_keys

    """
    Test Chat Completion API (streaming)
    """
    dial_configs = [
        {"quality": 0.5, "cost": 0.5},
    ]

    print()
    print("Testing Chat Completion Streaming")
    print("---------------------------------")
    for dials in dial_configs:
        dialtone = AsyncDialtone(
            api_key=api_keys["dialtone"],
            provider_config=ProviderConfig(
                openai=ProviderConfig.OpenAI(api_key=api_keys["openai"]),
                anthropic=ProviderConfig.Anthropic(api_key=api_keys["anthropic"]),
                google=ProviderConfig.Google(api_key=api_keys["google"]),
                groq=ProviderConfig.Groq(api_key=api_keys["groq"]),
                cohere=ProviderConfig.Cohere(api_key=api_keys["cohere"]),
                fireworks=ProviderConfig.Fireworks(api_key=api_keys["fireworks"]),
                deepinfra=ProviderConfig.DeepInfra(api_key=api_keys["deepinfra"]),
                together=ProviderConfig.Together(api_key=api_keys["together"]),
                replicate=ProviderConfig.Replicate(api_key=api_keys["replicate"]),
            ),
            router_model_config=RouterModelConfig(
                llama_3_70b=RouterModelConfig.Llama_3_70B(
                    no_tools_providers=[Provider.Fireworks, Provider.Groq],
                )
            ),
            dials=dials,
            base_url="http://localhost:8000",
        )

        response = await dialtone.chat.completions.create(
            messages=[
                ChatMessage(
                    role="user",
                    content="Hey, what's up? Add many newlines to your response.",
                )
            ],
            stream=True,
        )

        assert isinstance(response, AsyncGenerator)

        async for chat_completion_chunk in response:
            print(chat_completion_chunk)
            assert isinstance(chat_completion_chunk, ChatCompletionChunk)


@pytest.mark.asyncio
async def test_chat_route():
    global api_keys

    """
    Test Chat Completion API
    """
    dial_configs = [
        {"quality": 1, "cost": 0},
        {"quality": 0.75, "cost": 0.25},
        {"quality": 0.5, "cost": 0.5},
        {"quality": 0.25, "cost": 0.75},
        {"quality": 0, "cost": 1},
    ]

    print()
    print("Testing Chat Route")
    print("-----------------")
    for dials in dial_configs:
        dialtone = AsyncDialtone(
            api_key=api_keys["dialtone"],
            provider_config=ProviderConfig(
                openai=ProviderConfig.OpenAI(api_key=api_keys["openai"]),
                anthropic=ProviderConfig.Anthropic(api_key=api_keys["anthropic"]),
                google=ProviderConfig.Google(api_key=api_keys["google"]),
                groq=ProviderConfig.Groq(api_key=api_keys["groq"]),
                cohere=ProviderConfig.Cohere(api_key=api_keys["cohere"]),
                fireworks=ProviderConfig.Fireworks(api_key=api_keys["fireworks"]),
                deepinfra=ProviderConfig.DeepInfra(api_key=api_keys["deepinfra"]),
                together=ProviderConfig.Together(api_key=api_keys["together"]),
                replicate=ProviderConfig.Replicate(api_key=api_keys["replicate"]),
            ),
            router_model_config=RouterModelConfig(
                llama_3_70b=RouterModelConfig.Llama_3_70B(
                    no_tools_providers=[Provider.Fireworks, Provider.Groq],
                )
            ),
            dials=dials,
            base_url="http://localhost:8000",
        )

        response = await dialtone.chat.route(
            messages=[ChatMessage(role="user", content="Hello, world!")]
        )

        assert response.model is not None
        assert response.quality_predictions is not None
        assert response.routing_strategy is not None
        assert response.providers is not None

        print(f"Dials: {dials} | Model: {response.model}")


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": unit}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


@pytest.mark.asyncio
async def test_all_model_provider_combinations():
    global api_keys

    """
    Test All Model Provider Combinations
    """
    all_model_provider_combinations = {
        "openai": {
            "provider_config": ProviderConfig(
                openai=ProviderConfig.OpenAI(api_key=api_keys["openai"]),
            ),
            "model_configs": {
                "gpt_4o": RouterModelConfig(
                    include_models=[LLM.gpt_4o], gpt_4o=RouterModelConfig.OpenAI()
                ),
                "gpt_4o_mini": RouterModelConfig(
                    include_models=[LLM.gpt_4o_mini],
                    gpt_4o_mini=RouterModelConfig.OpenAI(),
                ),
            },
        },
        "anthropic": {
            "provider_config": ProviderConfig(
                anthropic=ProviderConfig.Anthropic(api_key=api_keys["anthropic"]),
            ),
            "model_configs": {
                "claude_3_5_sonnet": RouterModelConfig(
                    include_models=[LLM.claude_3_5_sonnet],
                    claude_3_5_sonnet=RouterModelConfig.Anthropic(),
                ),
                "claude_3_haiku": RouterModelConfig(
                    include_models=[LLM.claude_3_haiku],
                    claude_3_haiku=RouterModelConfig.Anthropic(),
                ),
            },
        },
        "google": {
            "provider_config": ProviderConfig(
                google=ProviderConfig.Google(api_key=api_keys["google"]),
            ),
            "model_configs": {
                "gemini_1_5_pro": RouterModelConfig(
                    include_models=[LLM.gemini_1_5_pro],
                    gemini_1_5_pro=RouterModelConfig.Google(),
                ),
                "gemini_1_5_flash": RouterModelConfig(
                    include_models=[LLM.gemini_1_5_flash],
                    gemini_1_5_flash=RouterModelConfig.Google(),
                ),
            },
        },
        "groq": {
            "provider_config": ProviderConfig(
                groq=ProviderConfig.Groq(api_key=api_keys["groq"]),
            ),
            "model_configs": {
                "llama_3_70b": RouterModelConfig(
                    include_models=[LLM.llama_3_70b],
                    llama_3_70b=RouterModelConfig.Llama_3_70B(
                        tools_providers=[Provider.Groq],
                        no_tools_providers=[Provider.Groq],
                    ),
                ),
                "llama_3_1_8b": RouterModelConfig(
                    include_models=[LLM.llama_3_1_8b],
                    llama_3_1_8b=RouterModelConfig.Llama_3_1_8B(
                        tools_providers=[Provider.Groq],
                        no_tools_providers=[Provider.Groq],
                    ),
                ),
                "llama_3_1_70b": RouterModelConfig(
                    include_models=[LLM.llama_3_1_70b],
                    llama_3_1_70b=RouterModelConfig.Llama_3_1_70B(
                        tools_providers=[Provider.Groq],
                        no_tools_providers=[Provider.Groq],
                    ),
                ),
                # TODO: Add llama_3_1_405b once groq default supports it for all users
            },
        },
        "cohere": {
            "provider_config": ProviderConfig(
                cohere=ProviderConfig.Cohere(api_key=api_keys["cohere"]),
            ),
            "model_configs": {
                "command_r_plus": RouterModelConfig(
                    include_models=[LLM.command_r_plus],
                    command_r_plus=RouterModelConfig.Cohere(),
                ),
                "command_r": RouterModelConfig(
                    include_models=[LLM.command_r], command_r=RouterModelConfig.Cohere()
                ),
            },
        },
        "fireworks": {
            "provider_config": ProviderConfig(
                fireworks=ProviderConfig.Fireworks(api_key=api_keys["fireworks"]),
            ),
            "model_configs": {
                "llama_3_70b": RouterModelConfig(
                    include_models=[LLM.llama_3_70b],
                    llama_3_70b=RouterModelConfig.Llama_3_70B(
                        no_tools_providers=[Provider.Fireworks],
                    ),
                ),
                "llama_3_1_8b": RouterModelConfig(
                    include_models=[LLM.llama_3_1_8b],
                    llama_3_1_8b=RouterModelConfig.Llama_3_1_8B(
                        no_tools_providers=[Provider.Fireworks],
                    ),
                ),
                "llama_3_1_70b": RouterModelConfig(
                    include_models=[LLM.llama_3_1_70b],
                    llama_3_1_70b=RouterModelConfig.Llama_3_1_70B(
                        no_tools_providers=[Provider.Fireworks],
                    ),
                ),
                "llama_3_1_405b": RouterModelConfig(
                    include_models=[LLM.llama_3_1_405b],
                    llama_3_1_405b=RouterModelConfig.Llama_3_1_405B(
                        no_tools_providers=[Provider.Fireworks],
                    ),
                ),
            },
        },
        "deepinfra": {
            "provider_config": ProviderConfig(
                deepinfra=ProviderConfig.DeepInfra(api_key=api_keys["deepinfra"]),
            ),
            "model_configs": {
                "llama_3_70b": RouterModelConfig(
                    include_models=[LLM.llama_3_70b],
                    llama_3_70b=RouterModelConfig.Llama_3_70B(
                        no_tools_providers=[Provider.DeepInfra],
                    ),
                ),
                "llama_3_1_8b": RouterModelConfig(
                    include_models=[LLM.llama_3_1_8b],
                    llama_3_1_8b=RouterModelConfig.Llama_3_1_8B(
                        no_tools_providers=[Provider.DeepInfra],
                    ),
                ),
                "llama_3_1_70b": RouterModelConfig(
                    include_models=[LLM.llama_3_1_70b],
                    llama_3_1_70b=RouterModelConfig.Llama_3_1_70B(
                        no_tools_providers=[Provider.DeepInfra],
                    ),
                ),
                "llama_3_1_405b": RouterModelConfig(
                    include_models=[LLM.llama_3_1_405b],
                    llama_3_1_405b=RouterModelConfig.Llama_3_1_405B(
                        no_tools_providers=[Provider.DeepInfra],
                    ),
                ),
            },
        },
        "together": {
            "provider_config": ProviderConfig(
                together=ProviderConfig.Together(api_key=api_keys["together"]),
            ),
            "model_configs": {
                "llama_3_70b": RouterModelConfig(
                    include_models=[LLM.llama_3_70b],
                    llama_3_70b=RouterModelConfig.Llama_3_70B(
                        no_tools_providers=[Provider.Together],
                    ),
                ),
                "llama_3_1_8b": RouterModelConfig(
                    include_models=[LLM.llama_3_1_8b],
                    llama_3_1_8b=RouterModelConfig.Llama_3_1_8B(
                        no_tools_providers=[Provider.Together],
                    ),
                ),
                "llama_3_1_70b": RouterModelConfig(
                    include_models=[LLM.llama_3_1_70b],
                    llama_3_1_70b=RouterModelConfig.Llama_3_1_70B(
                        no_tools_providers=[Provider.Together],
                    ),
                ),
                "llama_3_1_405b": RouterModelConfig(
                    include_models=[LLM.llama_3_1_405b],
                    llama_3_1_405b=RouterModelConfig.Llama_3_1_405B(
                        no_tools_providers=[Provider.Together],
                    ),
                ),
            },
        },
        "replicate": {
            "provider_config": ProviderConfig(
                replicate=ProviderConfig.Replicate(api_key=api_keys["replicate"]),
            ),
            "model_configs": {
                "llama_3_70b": RouterModelConfig(
                    include_models=[LLM.llama_3_70b],
                    llama_3_70b=RouterModelConfig.Llama_3_70B(
                        no_tools_providers=[Provider.Replicate],
                    ),
                ),
            },
        },
    }

    print()
    print("Testing All Model Provider Combinations")
    print("---------------------------------------")
    for provider_name, configs in all_model_provider_combinations.items():
        provider_config = configs["provider_config"]
        for model_name, model_config in configs["model_configs"].items():
            print(f"Testing | Provider: {provider_name} | Model: {model_name}")
            dialtone = AsyncDialtone(
                api_key=api_keys["dialtone"],
                provider_config=provider_config,
                router_model_config=model_config,
                base_url="http://localhost:8000",
            )

            response = await dialtone.chat.completions.create(
                messages=[ChatMessage(role="user", content="Hello, world!")]
            )

            assert isinstance(response, ChatCompletion)

            choice = response.choices[0]
            assert choice.message.content is not None
            assert choice.message.role == "assistant"
            assert response.model is not None
            assert response.provider is not None

            usage = response.usage
            assert usage.prompt_tokens is not None
            assert usage.completion_tokens is not None
            assert usage.total_tokens is not None


@pytest.mark.asyncio
async def test_all_model_provider_combinations_with_tools():
    global api_keys

    """
    Test all model provider combinations with tools
    """
    all_model_provider_combinations = {
        "openai": {
            "provider_config": ProviderConfig(
                openai=ProviderConfig.OpenAI(api_key=api_keys["openai"]),
            ),
            "model_configs": {
                "gpt_4o": RouterModelConfig(
                    include_models=[LLM.gpt_4o],
                    gpt_4o=RouterModelConfig.OpenAI(),
                ),
                "gpt_4o_mini": RouterModelConfig(
                    include_models=[LLM.gpt_4o_mini],
                    gpt_4o_mini=RouterModelConfig.OpenAI(),
                ),
            },
        },
        "anthropic": {
            "provider_config": ProviderConfig(
                anthropic=ProviderConfig.Anthropic(api_key=api_keys["anthropic"]),
            ),
            "model_configs": {
                "claude_3_5_sonnet": RouterModelConfig(
                    include_models=[LLM.claude_3_5_sonnet],
                    claude_3_5_sonnet=RouterModelConfig.Anthropic(),
                ),
                "claude_3_haiku": RouterModelConfig(
                    include_models=[LLM.claude_3_haiku],
                    claude_3_haiku=RouterModelConfig.Anthropic(),
                ),
            },
        },
        "google": {
            "provider_config": ProviderConfig(
                google=ProviderConfig.Google(api_key=api_keys["google"]),
            ),
            "model_configs": {
                "gemini_1_5_pro": RouterModelConfig(
                    include_models=[LLM.gemini_1_5_pro],
                    gemini_1_5_pro=RouterModelConfig.Google(),
                ),
                "gemini_1_5_flash": RouterModelConfig(
                    include_models=[LLM.gemini_1_5_flash],
                    gemini_1_5_flash=RouterModelConfig.Google(),
                ),
            },
        },
        "groq": {
            "provider_config": ProviderConfig(
                groq=ProviderConfig.Groq(api_key=api_keys["groq"]),
            ),
            "model_configs": {
                "llama_3_70b": RouterModelConfig(
                    include_models=[LLM.llama_3_70b],
                    llama_3_70b=RouterModelConfig.Llama_3_70B(
                        tools_providers=[Provider.Groq],
                        no_tools_providers=[Provider.Groq],
                    ),
                ),
                "llama_3_1_8b": RouterModelConfig(
                    include_models=[LLM.llama_3_1_8b],
                    llama_3_1_8b=RouterModelConfig.Llama_3_1_8B(
                        tools_providers=[Provider.Groq],
                        no_tools_providers=[Provider.Groq],
                    ),
                ),
                "llama_3_1_70b": RouterModelConfig(
                    include_models=[LLM.llama_3_1_70b],
                    llama_3_1_70b=RouterModelConfig.Llama_3_1_70B(
                        tools_providers=[Provider.Groq],
                        no_tools_providers=[Provider.Groq],
                    ),
                ),
                # TODO: Add llama_3_1_405b once groq default supports it for all users
            },
        },
        "cohere": {
            "provider_config": ProviderConfig(
                cohere=ProviderConfig.Cohere(api_key=api_keys["cohere"]),
            ),
            "model_configs": {
                "command_r_plus": RouterModelConfig(
                    include_models=[LLM.command_r_plus],
                    command_r_plus=RouterModelConfig.Cohere(),
                ),
                "command_r": RouterModelConfig(
                    include_models=[LLM.command_r],
                    command_r=RouterModelConfig.Cohere(),
                ),
            },
        },
        "deepinfra": {
            "provider_config": ProviderConfig(
                deepinfra=ProviderConfig.DeepInfra(api_key=api_keys["deepinfra"]),
            ),
            "model_configs": {
                "llama_3_70b": RouterModelConfig(
                    include_models=[LLM.llama_3_70b],
                    llama_3_70b=RouterModelConfig.Llama_3_70B(
                        tools_providers=[Provider.DeepInfra],
                        no_tools_providers=[Provider.DeepInfra],
                    ),
                ),
            },
        },
    }

    print()
    print("Testing all model provider combinations with tools")
    print("--------------------------------------------------")
    for provider_name, configs in all_model_provider_combinations.items():
        provider_config = configs["provider_config"]
        for model_name, model_config in configs["model_configs"].items():
            print(f"Testing | Provider: {provider_name} | Model: {model_name}")
            dialtone = AsyncDialtone(
                api_key=api_keys["dialtone"],
                provider_config=provider_config,
                router_model_config=model_config,
                base_url="http://localhost:8000",
            )

            messages = [
                {
                    "role": "user",
                    "content": "What's the weather like in San Francisco in fahrenheit?",
                }
            ]
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location"],
                        },
                    },
                }
            ]

            response = await dialtone.chat.completions.create(
                messages=messages,
                tools=tools,
            )

            assert isinstance(response, ChatCompletion)

            choice = response.choices[0]
            assert choice.message.content is not None
            assert choice.message.role == "assistant"
            assert response.model is not None
            assert response.provider is not None

            usage = response.usage
            assert usage.prompt_tokens is not None
            assert usage.completion_tokens is not None
            assert usage.total_tokens is not None

            print("first response", response)

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            assert tool_calls is not None and len(tool_calls) > 0

            available_functions = {
                "get_current_weather": get_current_weather,
            }  # only one function in this example, but you can have multiple
            messages.append(
                response_message.__dict__
            )  # extend conversation with assistant's reply

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"),
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response
            second_response = await dialtone.chat.completions.create(
                messages=messages,
                tools=tools,
            )  # get a new response from the model where it can see the function response

            print("second response", second_response)

            assert isinstance(second_response, ChatCompletion)

            choice = second_response.choices[0]
            assert choice.message.content is not None
            assert choice.message.role == "assistant"
            assert second_response.model is not None
            assert second_response.provider is not None

            usage = second_response.usage
            assert usage.prompt_tokens is not None
            assert usage.completion_tokens is not None
            assert usage.total_tokens is not None
