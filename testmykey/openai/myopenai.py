#-----------------OPRNAI_ZL-----------------##
# python3
# please install OpenAI SDK first: `pip3 install openai`
from openai import OpenAI

client = OpenAI(api_key="<sk-zKs3eVUnmpT281l1bs6CdFPUaabv0ocMSonmH9FekGftF5hZ>", base_url="https://api.chatanywhere.tech")

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]
)

print(response.choices[0].message.content)


# ##-----------------OPRNAI_TB-----------------##
# # python3
# # please install OpenAI SDK first: `pip3 install openai`
# from openai import OpenAI

# client = OpenAI(api_key="<sk-iV0XN5ue0PHE6mhR612491C1F3Cf4562B35dB291C5Dd329a>", base_url="https://api.pumpkinaigc.online")

# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "你是谁"},
#     ]
# )
# print(response)

# # print(response.choices[0].message.content)