# # bash
# curl https://api.deepseek.com/chat/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
#   -d '{
#         "model": "deepseek-chat",
#         "messages": [
#           {"role": "system", "content": "You are a helpful assistant."},
#           {"role": "user", "content": "Hello!"}
#         ]
#       }'


# curl https://api.chatanywhere.tech/v1/chat/completions \
#   -H 'Content-Type: application/json' \
#   -H 'Authorization: Bearer sk-1WSYBg4Sg8AwZu0Zr1mu1eTZmnCsFDuoq4sSHZMgEZbEt5iE' \
#   -d '{
#         "model": "gpt-3.5-turbo",
#         "messages": [{"role": "user", "content": "Say this is a test!"}],
#         "temperature": 0.7
#         }'

curl https://api.chatanywhere.tech/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-zKs3eVUnmpT281l1bs6CdFPUaabv0ocMSonmH9FekGftF5hZ' \
  -d '{
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Say this is a test!"}],
        "temperature": 0.7
        }'


# curl https://api.chatanywhere.tech/v1/chat/completions \
#   -H 'Content-Type: application/json' \
#   -H 'Authorization: Bearer  sk-6ien0ZfxkXYW4a4kKI4rkhmbWyQD3b3SXeZmo6gfwuf1VN4U' \
#   -d '{
#         "model": "gpt-3.5-turbo",
#         "messages": [{"role": "user", "content": "Say this is a test!"}],
#         "temperature": 0.7
#         }'

# curl https://api.chatanywhere.tech/v1/chat/completions \
#   -H 'Content-Type: application/json' \
#   -H 'Authorization: Bearer  sk-MF9UsaCNeOhVCdLuw2QGHgj3wdloSr507TQLEU34HtRbgYHM' \
#   -d '{
#         "model": "gpt-3.5-turbo",
#         "messages": [{"role": "user", "content": "Say this is a test!"}],
#         "temperature": 0.7
#         }'