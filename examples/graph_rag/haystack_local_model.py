import torch

from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack.utils import Secret, ComponentDevice, Device

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from haystack.dataclasses import ChatMessage

messages = [
    {"role": "system", "content": """
A chat between a curious user and an artificial intelligence Assistant. The Assistant is an expert at identifying entities and relationships in text. The Assistant responds in JSON output only.

The User provides text in the format:

-------Text begin-------
<User provided text>
-------Text end-------

The Assistant follows the following steps before replying to the User:

1. **identify the most important entities** The Assistant identifies the most important entities in the text. These entities are listed in the JSON output under the key "nodes", they follow the structure of a list of dictionaries where each dict is:

"nodes":[{"id": <entity N>, "type": <type>, "detailed_type": <detailed type>}, ...]

where "type": <type> is a broad categorization of the entity. "detailed type": <detailed_type>  is a very descriptive categorization of the entity.

2. **determine relationships** The Assistant uses the text between -------Text begin------- and -------Text end------- to determine the relationships between the entities identified in the "nodes" list defined above. These relationships are called "edges" and they follow the structure of:

"edges":[{"from": <entity 1>, "to": <entity 2>, "label": <relationship>}, ...]

The <entity N> must correspond to the "id" of an entity in the "nodes" list.

The Assistant never repeats the same node twice. The Assistant never repeats the same edge twice.
The Assistant responds to the User in JSON only, according to the following JSON schema:

{"type":"object","properties":{"nodes":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"type":{"type":"string"},"detailed_type":{"type":"string"}},"required":["id","type","detailed_type"],"additionalProperties":false}},"edges":{"type":"array","items":{"type":"object","properties":{"from":{"type":"string"},"to":{"type":"string"},"label":{"type":"string"}},"required":["from","to","label"],"additionalProperties":false}}},"required":["nodes","edges"],"additionalProperties":false}
     """},
    {"role": "user", "content": """
-------Text begin-------
OpenAI is an American artificial intelligence (AI) research organization founded in December 2015 and headquartered in San Francisco, California. Its mission is to develop "safe and beneficial" artificial general intelligence, which it defines as "highly autonomous systems that outperform humans at most economically valuable work".[4] As a leading organization in the ongoing AI boom,[5] OpenAI is known for the GPT family of large language models, the DALL-E series of text-to-image models, and a text-to-video model named Sora.[6][7] Its release of ChatGPT in November 2022 has been credited with catalyzing widespread interest in generative AI.
-------Text end-------
"""}
]


def huggingface():
    torch.random.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        "EmergentMethods/Phi-3-mini-4k-instruct-graph",
        device_map="mps",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("EmergentMethods/Phi-3-mini-4k-instruct-graph")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    output = pipe(messages, **generation_args)
    return output[0]['generated_text']


def haystack():
    generator = HuggingFaceLocalChatGenerator(
        token=Secret.from_token("HF_API_TOKEN"),
        task="text-generation",
        model="EmergentMethods/Phi-3-mini-4k-instruct-graph",
        device=ComponentDevice.from_single(Device.mps()),
        generation_kwargs = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False
        }
    )

    generator.warm_up()
    messages[0]['name'] = "foo"
    messages[1]['name'] = "bar"

    chat_messages = [ChatMessage.from_dict(messages[0]), ChatMessage.from_dict(messages[1])]
    output = generator.run(chat_messages)

    return output['replies'][0]


def main():
    output_original = huggingface()
    output_haystack = haystack()

    print("Original output:")
    print(output_original)
    print("\n\n\n")
    print("Haystack output:")
    print(output_haystack)


if __name__ == '__main__':
    main()